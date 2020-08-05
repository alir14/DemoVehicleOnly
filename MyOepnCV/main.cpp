// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include <cldnn/cldnn_config.hpp>
#include <inference_engine.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/args_helper.hpp>

#include "common.hpp"
#include "grid_mat.hpp"
#include "input_wrappers.hpp"
#include "security_barrier_camera_demo.hpp"
#include "net_wrappers.hpp"

using namespace InferenceEngine;

typedef std::chrono::duration<float, std::chrono::seconds::period> Sec;

struct BboxAndDescr {
    enum class ObjectType {
        NONE,
        VEHICLE,
        PLATE,
    } objectType;
    cv::Rect rect;
    std::string descr;
};

struct InferRequestsContainer {
    InferRequestsContainer() = default;
    InferRequestsContainer(const InferRequestsContainer&) = delete;
    InferRequestsContainer& operator=(const InferRequestsContainer&) = delete;

    void assign(const std::vector<InferRequest>& inferRequests) {
        actualInferRequests = inferRequests;
        this->inferRequests.container.clear();

        for (auto& ir : this->actualInferRequests) {
            this->inferRequests.container.push_back(ir);
        }
    }

    std::vector<InferRequest> getActualInferRequests() {
        return actualInferRequests;
    }
    ConcurrentContainer<std::vector<std::reference_wrapper<InferRequest>>> inferRequests;

private:
    std::vector<InferRequest> actualInferRequests;
};

struct Context {  // stores all global data for tasks
    Context(const std::vector<std::shared_ptr<InputChannel>>& inputChannels,
        const Detector& detector,
        const VehicleAttributesClassifier& vehicleAttributesClassifier, const Lpr& lpr,
        int pause, const std::vector<cv::Size>& gridParam, cv::Size displayResolution, std::chrono::steady_clock::duration showPeriod,
        const std::string& monitorsStr,
        uint64_t lastFrameId,
        uint64_t nireq,
        bool isVideo,
        std::size_t nclassifiersireq, std::size_t nrecognizersireq) :
        readersContext{ inputChannels, std::vector<int64_t>(inputChannels.size(), -1), std::vector<std::mutex>(inputChannels.size()) },
        inferTasksContext{ detector },
        detectionsProcessorsContext{ vehicleAttributesClassifier, lpr },
        drawersContext{ pause, gridParam, displayResolution, showPeriod, monitorsStr },
        videoFramesContext{ std::vector<uint64_t>(inputChannels.size(), lastFrameId), std::vector<std::mutex>(inputChannels.size()) },
        nireq{ nireq },
        isVideo{ isVideo },
        t0{ std::chrono::steady_clock::time_point() },
        freeDetectionInfersCount{ 0 },
        frameCounter{ 0 }
    {
        assert(inputChannels.size() == gridParam.size());
        std::vector<InferRequest> detectorInferRequests;
        std::vector<InferRequest> attributesInferRequests;
        std::vector<InferRequest> lprInferRequests;
        detectorInferRequests.reserve(nireq);
        attributesInferRequests.reserve(nclassifiersireq);
        lprInferRequests.reserve(nrecognizersireq);
        std::generate_n(std::back_inserter(detectorInferRequests), nireq, [&] {
            return inferTasksContext.detector.createInferRequest(); });
        std::generate_n(std::back_inserter(attributesInferRequests), nclassifiersireq, [&] {
            return detectionsProcessorsContext.vehicleAttributesClassifier.createInferRequest(); });
        std::generate_n(std::back_inserter(lprInferRequests), nrecognizersireq, [&] {
            return detectionsProcessorsContext.lpr.createInferRequest(); });
        detectorsInfers.assign(detectorInferRequests);
        attributesInfers.assign(attributesInferRequests);
        platesInfers.assign(lprInferRequests);
    }
    struct {
        std::vector<std::shared_ptr<InputChannel>> inputChannels;
        std::vector<int64_t> lastCapturedFrameIds;
        std::vector<std::mutex> lastCapturedFrameIdsMutexes;
        std::weak_ptr<Worker> readersWorker;
    } readersContext;
    struct {
        Detector detector;
        std::weak_ptr<Worker> inferTasksWorker;
    } inferTasksContext;
    struct {
        VehicleAttributesClassifier vehicleAttributesClassifier;
        Lpr lpr;
        std::weak_ptr<Worker> detectionsProcessorsWorker;
    } detectionsProcessorsContext;
    struct DrawersContext {
        DrawersContext(int pause, const std::vector<cv::Size>& gridParam, cv::Size displayResolution, std::chrono::steady_clock::duration showPeriod,
            const std::string& monitorsStr) :
            pause{ pause }, gridParam{ gridParam }, displayResolution{ displayResolution }, showPeriod{ showPeriod },
            lastShownframeId{ 0 }, prevShow{ std::chrono::steady_clock::time_point() }, framesAfterUpdate{ 0 }, updateTime{ std::chrono::steady_clock::time_point() },
            presenter{ monitorsStr,
                GridMat(gridParam, displayResolution).outimg.rows - 70,
                cv::Size{GridMat(gridParam, displayResolution).outimg.cols / 4, 60} } {}
        int pause;
        std::vector<cv::Size> gridParam;
        cv::Size displayResolution;
        std::chrono::steady_clock::duration showPeriod;  // desiered frequency of imshow
        std::weak_ptr<Worker> drawersWorker;
        int64_t lastShownframeId;
        std::chrono::steady_clock::time_point prevShow;  // time stamp of previous imshow
        std::map<int64_t, GridMat> gridMats;
        std::mutex drawerMutex;
        std::ostringstream outThroughput;
        unsigned framesAfterUpdate;
        std::chrono::steady_clock::time_point updateTime;
        Presenter presenter;
    } drawersContext;
    struct {
        std::vector<uint64_t> lastframeIds;
        std::vector<std::mutex> lastFrameIdsMutexes;
    } videoFramesContext;
    std::weak_ptr<Worker> resAggregatorsWorker;
    std::mutex classifiersAggreagatorPrintMutex;
    uint64_t nireq;
    bool isVideo;
    std::chrono::steady_clock::time_point t0;
    std::atomic<std::vector<InferRequest>::size_type> freeDetectionInfersCount;
    std::atomic<uint64_t> frameCounter;
    InferRequestsContainer detectorsInfers, attributesInfers, platesInfers;
};

class ReborningVideoFrame : public VideoFrame {
public:
    ReborningVideoFrame(Context& context, const unsigned sourceID, const int64_t frameId, const cv::Mat& frame = cv::Mat()) :
        VideoFrame{ sourceID, frameId, frame }, context(context) {}  // can not write context{context} because of CentOS 7.4 compiler bug
    virtual ~ReborningVideoFrame();
    Context& context;
};

class Drawer : public Task {  // accumulates and shows processed frames
public:
    explicit Drawer(VideoFrame::Ptr sharedVideoFrame) :
        Task{ sharedVideoFrame, 1.0 } {}
    bool isReady() override;
    void process() override;
};

class ResAggregator : public Task {  // draws results on the frame
public:
    ResAggregator(const VideoFrame::Ptr& sharedVideoFrame, std::list<BboxAndDescr>&& boxesAndDescrs) :
        Task{ sharedVideoFrame, 4.0 }, boxesAndDescrs{ std::move(boxesAndDescrs) } {}
    bool isReady() override {
        return true;
    }
    void process() override;
private:
    std::list<BboxAndDescr> boxesAndDescrs;
};

class ClassifiersAggreagator {  // waits for all classifiers and recognisers accumulating results
public:
    std::string rawDetections;
    ConcurrentContainer<std::list<std::string>> rawAttributes;
    ConcurrentContainer<std::list<std::string>> rawDecodedPlates;

    explicit ClassifiersAggreagator(const VideoFrame::Ptr& sharedVideoFrame) :
        sharedVideoFrame{ sharedVideoFrame } {}
    ~ClassifiersAggreagator() {
        std::mutex& printMutex = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.classifiersAggreagatorPrintMutex;
        printMutex.lock();
        std::cout << rawDetections;
        for (const std::string& rawAttribute : rawAttributes.container) {  // destructor assures that none uses the container
            std::cout << rawAttribute;
        }
        for (const std::string& rawDecodedPlate : rawDecodedPlates.container) {
            std::cout << rawDecodedPlate;
        }
        printMutex.unlock();
        tryPush(static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.resAggregatorsWorker,
            std::make_shared<ResAggregator>(sharedVideoFrame, std::move(boxesAndDescrs)));
    }
    void push(BboxAndDescr&& bboxAndDescr) {
        boxesAndDescrs.lockedPush_back(std::move(bboxAndDescr));
    }
    const VideoFrame::Ptr sharedVideoFrame;

private:
    ConcurrentContainer<std::list<BboxAndDescr>> boxesAndDescrs;
};

class DetectionsProcessor : public Task {  // extracts detections from blob InferRequests and runs classifiers and recognisers
public:
    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame, InferRequest* inferRequest) :
        Task{ sharedVideoFrame, 1.0 }, inferRequest{ inferRequest }, requireGettingNumberOfDetections{ true } {}
    DetectionsProcessor(VideoFrame::Ptr sharedVideoFrame, std::shared_ptr<ClassifiersAggreagator>&& classifiersAggreagator, std::list<cv::Rect>&& vehicleRects,
        std::list<cv::Rect>&& plateRects) :
        Task{ sharedVideoFrame, 1.0 }, classifiersAggreagator{ std::move(classifiersAggreagator) }, inferRequest{ nullptr },
        vehicleRects{ std::move(vehicleRects) }, plateRects{ std::move(plateRects) }, requireGettingNumberOfDetections{ false } {}
    bool isReady() override;
    void process() override;

private:
    std::shared_ptr<ClassifiersAggreagator> classifiersAggreagator;  // when no one stores this object we will draw
    InferRequest* inferRequest;
    std::list<cv::Rect> vehicleRects;
    std::list<cv::Rect> plateRects;
    std::vector<std::reference_wrapper<InferRequest>> reservedAttributesRequests;
    std::vector<std::reference_wrapper<InferRequest>> reservedLprRequests;
    bool requireGettingNumberOfDetections;
};

class InferTask : public Task {  // runs detection
public:
    explicit InferTask(VideoFrame::Ptr sharedVideoFrame) :
        Task{ sharedVideoFrame, 5.0 } {}
    bool isReady() override;
    void process() override;
};

class Reader : public Task {
public:
    explicit Reader(VideoFrame::Ptr sharedVideoFrame) :
        Task{ sharedVideoFrame, 2.0 } {}
    bool isReady() override;
    void process() override;
};

ReborningVideoFrame::~ReborningVideoFrame() {
    try {
        const std::shared_ptr<Worker>& worker = std::shared_ptr<Worker>(context.readersContext.readersWorker);
        context.videoFramesContext.lastFrameIdsMutexes[sourceID].lock();
        const auto frameId = ++context.videoFramesContext.lastframeIds[sourceID];
        context.videoFramesContext.lastFrameIdsMutexes[sourceID].unlock();
        std::shared_ptr<ReborningVideoFrame> reborn = std::make_shared<ReborningVideoFrame>(context, sourceID, frameId, frame);
        worker->push(std::make_shared<Reader>(reborn));
    }
    catch (const std::bad_weak_ptr&) {}
}

bool Drawer::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    std::chrono::steady_clock::time_point prevShow = context.drawersContext.prevShow;
    std::chrono::steady_clock::duration showPeriod = context.drawersContext.showPeriod;
    if (1u == context.drawersContext.gridParam.size()) {
        if (std::chrono::steady_clock::now() - prevShow > showPeriod) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        std::map<int64_t, GridMat>& gridMats = context.drawersContext.gridMats;
        auto gridMatIt = gridMats.find(sharedVideoFrame->frameId);
        if (gridMats.end() == gridMatIt) {
            if (2 > gridMats.size()) {  // buffer size
                return true;
            }
            else {
                return false;
            }
        }
        else {
            if (1u == gridMatIt->second.getUnupdatedSourceIDs().size()) {
                if (context.drawersContext.lastShownframeId == sharedVideoFrame->frameId
                    && std::chrono::steady_clock::now() - prevShow > showPeriod) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else {
                return true;
            }
        }
    }
}

void Drawer::process() {
    const int64_t frameId = sharedVideoFrame->frameId;
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    std::map<int64_t, GridMat>& gridMats = context.drawersContext.gridMats;
    context.drawersContext.drawerMutex.lock();
    auto gridMatIt = gridMats.find(frameId);
    if (gridMats.end() == gridMatIt) {
        gridMatIt = gridMats.emplace(frameId, GridMat(context.drawersContext.gridParam,
            context.drawersContext.displayResolution)).first;
    }

    gridMatIt->second.update(sharedVideoFrame->frame, sharedVideoFrame->sourceID);
    auto firstGridIt = gridMats.begin();
    int64_t& lastShownframeId = context.drawersContext.lastShownframeId;
    if (firstGridIt->first == lastShownframeId && firstGridIt->second.isFilled()) {
        lastShownframeId++;
        cv::Mat mat = firstGridIt->second.getMat();

        constexpr float OPACITY = 0.6f;
        fillROIColor(mat, cv::Rect(5, 5, 390, 115), cv::Scalar(255, 0, 0), OPACITY);
        cv::putText(mat, "Detection InferRequests usage", cv::Point2f(15, 70), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 255, 255, 255 });
        cv::Rect usage(15, 90, 370, 20);
        cv::rectangle(mat, usage, { 0, 255, 0 }, 2);
        uint64_t nireq = context.nireq;
        uint64_t frameCounter = context.frameCounter;
        usage.width = static_cast<int>(usage.width * static_cast<float>(frameCounter * nireq - context.freeDetectionInfersCount) / (frameCounter * nireq));
        cv::rectangle(mat, usage, { 0, 255, 0 }, cv::FILLED);

        context.drawersContext.framesAfterUpdate++;
        const std::chrono::steady_clock::time_point localT1 = std::chrono::steady_clock::now();
        const Sec timeDuration = localT1 - context.drawersContext.updateTime;
        if (Sec{ 1 } <= timeDuration || context.drawersContext.updateTime == context.t0) {
            context.drawersContext.outThroughput.str("");
            context.drawersContext.outThroughput << std::fixed << std::setprecision(1)
                << static_cast<float>(context.drawersContext.framesAfterUpdate) / timeDuration.count() << "FPS";
            context.drawersContext.framesAfterUpdate = 0;
            context.drawersContext.updateTime = localT1;
        }

        cv::putText(mat, context.drawersContext.outThroughput.str(), cv::Point2f(15, 35), cv::FONT_HERSHEY_TRIPLEX, 0.7, cv::Scalar{ 255, 255, 255 });

        context.drawersContext.presenter.drawGraphs(mat);

        cv::imshow("Detection results", firstGridIt->second.getMat());
        context.drawersContext.prevShow = std::chrono::steady_clock::now();
        const int key = cv::waitKey(context.drawersContext.pause);
        if (key == 27 || 'q' == key || 'Q' == key || !context.isVideo) {
            try {
                std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
            }
            catch (const std::bad_weak_ptr&) {}
        }
        else if (key == 32) {
            context.drawersContext.pause = (context.drawersContext.pause + 1) & 1;
        }
        else {
            context.drawersContext.presenter.handleKey(key);
        }
        firstGridIt->second.clear();
        gridMats.emplace((--gridMats.end())->first + 1, firstGridIt->second);
        gridMats.erase(firstGridIt);
    }
    context.drawersContext.drawerMutex.unlock();
}

void ResAggregator::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    context.freeDetectionInfersCount += context.detectorsInfers.inferRequests.lockedSize();
    context.frameCounter++;
    if (!FLAGS_no_show) {
        for (const BboxAndDescr& bboxAndDescr : boxesAndDescrs) {
            switch (bboxAndDescr.objectType) {
            case BboxAndDescr::ObjectType::NONE: cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, { 255, 255, 0 }, 4);
                break;
            case BboxAndDescr::ObjectType::VEHICLE: cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, { 0, 255, 0 }, 4);
                cv::putText(sharedVideoFrame->frame, bboxAndDescr.descr,
                    cv::Point{ bboxAndDescr.rect.x, bboxAndDescr.rect.y + 35 },
                    cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 255, 0), 4);
                break;
            case BboxAndDescr::ObjectType::PLATE: cv::rectangle(sharedVideoFrame->frame, bboxAndDescr.rect, { 0, 0, 255 }, 4);
                cv::putText(sharedVideoFrame->frame, bboxAndDescr.descr,
                    cv::Point{ bboxAndDescr.rect.x, bboxAndDescr.rect.y - 10 },
                    cv::FONT_HERSHEY_COMPLEX, 1.3, cv::Scalar(0, 0, 255), 4);
                break;
            default: throw std::exception();  // must never happen
                break;
            }
        }
        tryPush(context.drawersContext.drawersWorker, std::make_shared<Drawer>(sharedVideoFrame));
    }
    else {
        if (!context.isVideo) {
            try {
                std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
            }
            catch (const std::bad_weak_ptr&) {}
        }
    }
}

bool DetectionsProcessor::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    if (requireGettingNumberOfDetections) {
        classifiersAggreagator = std::make_shared<ClassifiersAggreagator>(sharedVideoFrame);
        std::list<Detector::Result> results;
        if (!(true && ((sharedVideoFrame->frameId == 0 && !context.isVideo) || context.isVideo))) {
            results = context.inferTasksContext.detector.getResults(*inferRequest, sharedVideoFrame->frame.size());
        }
        else {
            std::ostringstream rawResultsStream;
            results = context.inferTasksContext.detector.getResults(*inferRequest, sharedVideoFrame->frame.size(), &rawResultsStream);
            classifiersAggreagator->rawDetections = rawResultsStream.str();
        }
        for (Detector::Result result : results) {
            switch (result.label) {
            case 1:
            {
                vehicleRects.emplace_back(result.location & cv::Rect{ cv::Point(0, 0), sharedVideoFrame->frame.size() });
                break;
            }
            case 2:
            {
                // expanding a bounding box a bit, better for the license plate recognition
                result.location.x -= 5;
                result.location.y -= 5;
                result.location.width += 10;
                result.location.height += 10;
                plateRects.emplace_back(result.location & cv::Rect{ cv::Point(0, 0), sharedVideoFrame->frame.size() });
                break;
            }
            default: throw std::exception();  // must never happen
                break;
            }
        }
        context.detectorsInfers.inferRequests.lockedPush_back(*inferRequest);
        requireGettingNumberOfDetections = false;
    }

    if ((vehicleRects.empty() || vehicleAttribute_Model.empty()) && (plateRects.empty() || plateLicense_Model.empty())) {
        return true;
    }
    else {
        // isReady() is called under mutexes so it is assured that available InferRequests will not be taken, but new InferRequests can come in
        // acquire as many InferRequests as it is possible or needed
        InferRequestsContainer& attributesInfers = context.attributesInfers;
        attributesInfers.inferRequests.mutex.lock();
        const std::size_t numberOfAttributesInferRequestsAcquired = std::min(vehicleRects.size(), attributesInfers.inferRequests.container.size());
        reservedAttributesRequests.assign(attributesInfers.inferRequests.container.end() - numberOfAttributesInferRequestsAcquired,
            attributesInfers.inferRequests.container.end());
        attributesInfers.inferRequests.container.erase(attributesInfers.inferRequests.container.end() - numberOfAttributesInferRequestsAcquired,
            attributesInfers.inferRequests.container.end());
        attributesInfers.inferRequests.mutex.unlock();

        InferRequestsContainer& platesInfers = context.platesInfers;
        platesInfers.inferRequests.mutex.lock();
        const std::size_t numberOfLprInferRequestsAcquired = std::min(plateRects.size(), platesInfers.inferRequests.container.size());
        reservedLprRequests.assign(platesInfers.inferRequests.container.end() - numberOfLprInferRequestsAcquired, platesInfers.inferRequests.container.end());
        platesInfers.inferRequests.container.erase(platesInfers.inferRequests.container.end() - numberOfLprInferRequestsAcquired,
            platesInfers.inferRequests.container.end());
        platesInfers.inferRequests.mutex.unlock();
        return numberOfAttributesInferRequestsAcquired || numberOfLprInferRequestsAcquired;
    }
}

void DetectionsProcessor::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;

    auto vehicleRectsIt = vehicleRects.begin();
    for (auto attributesRequestIt = reservedAttributesRequests.begin(); attributesRequestIt != reservedAttributesRequests.end();
        vehicleRectsIt++, attributesRequestIt++) {
        const cv::Rect vehicleRect = *vehicleRectsIt;
        InferRequest& attributesRequest = *attributesRequestIt;
        context.detectionsProcessorsContext.vehicleAttributesClassifier.setImage(attributesRequest, sharedVideoFrame->frame, vehicleRect);

        attributesRequest.SetCompletionCallback(
            std::bind(
                [](std::shared_ptr<ClassifiersAggreagator> classifiersAggreagator,
                    InferRequest& attributesRequest,
                    cv::Rect rect,
                    Context& context) {
                        attributesRequest.SetCompletionCallback([] {});  // destroy the stored bind object

                        const std::pair<std::string, std::string>& attributes
                            = context.detectionsProcessorsContext.vehicleAttributesClassifier.getResults(attributesRequest);

                        if (true && ((classifiersAggreagator->sharedVideoFrame->frameId == 0 && !context.isVideo) || context.isVideo)) { 
                            classifiersAggreagator->rawAttributes.lockedPush_back("Vehicle Attributes results:" + attributes.first + ';'
                                + attributes.second + '\n');
                        }
                        classifiersAggreagator->push(BboxAndDescr{ BboxAndDescr::ObjectType::VEHICLE, rect, attributes.first + ' ' + attributes.second });
                        context.attributesInfers.inferRequests.lockedPush_back(attributesRequest);
                }, classifiersAggreagator,
                std::ref(attributesRequest),
                    vehicleRect,
                    std::ref(context)));

        attributesRequest.StartAsync();
    }
    vehicleRects.erase(vehicleRects.begin(), vehicleRectsIt);

    auto plateRectsIt = plateRects.begin();
    for (auto lprRequestsIt = reservedLprRequests.begin(); lprRequestsIt != reservedLprRequests.end(); plateRectsIt++, lprRequestsIt++) {
        const cv::Rect plateRect = *plateRectsIt;
        InferRequest& lprRequest = *lprRequestsIt;
        context.detectionsProcessorsContext.lpr.setImage(lprRequest, sharedVideoFrame->frame, plateRect);

        lprRequest.SetCompletionCallback(
            std::bind(
                [](std::shared_ptr<ClassifiersAggreagator> classifiersAggreagator,
                    InferRequest& lprRequest,
                    cv::Rect rect,
                    Context& context) {
                        lprRequest.SetCompletionCallback([] {});  // destroy the stored bind object

                        std::string result = context.detectionsProcessorsContext.lpr.getResults(lprRequest);

                        if (true && ((classifiersAggreagator->sharedVideoFrame->frameId == 0 && !context.isVideo) || context.isVideo)) { 
                            classifiersAggreagator->rawDecodedPlates.lockedPush_back("License Plate Recognition results:" + result + '\n');
                        }
                        classifiersAggreagator->push(BboxAndDescr{ BboxAndDescr::ObjectType::PLATE, rect, std::move(result) });
                        context.platesInfers.inferRequests.lockedPush_back(lprRequest);
                }, classifiersAggreagator,
                std::ref(lprRequest),
                    plateRect,
                    std::ref(context)));

        lprRequest.StartAsync();
    }
    plateRects.erase(plateRects.begin(), plateRectsIt);
    

    if (!vehicleRects.empty() || !plateRects.empty()) {
        tryPush(context.detectionsProcessorsContext.detectionsProcessorsWorker,
            std::make_shared<DetectionsProcessor>(sharedVideoFrame, std::move(classifiersAggreagator), std::move(vehicleRects), std::move(plateRects)));
    }
}

bool InferTask::isReady() {
    InferRequestsContainer& detectorsInfers = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context.detectorsInfers;
    if (detectorsInfers.inferRequests.container.empty()) {
        return false;
    }
    else {
        detectorsInfers.inferRequests.mutex.lock();
        if (detectorsInfers.inferRequests.container.empty()) {
            detectorsInfers.inferRequests.mutex.unlock();
            return false;
        }
        else {
            return true;  // process() will unlock the mutex
        }
    }
}

void InferTask::process() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    InferRequestsContainer& detectorsInfers = context.detectorsInfers;
    std::reference_wrapper<InferRequest> inferRequest = detectorsInfers.inferRequests.container.back();
    detectorsInfers.inferRequests.container.pop_back();
    detectorsInfers.inferRequests.mutex.unlock();

    context.inferTasksContext.detector.setImage(inferRequest, sharedVideoFrame->frame);

    inferRequest.get().SetCompletionCallback(
        std::bind(
            [](VideoFrame::Ptr sharedVideoFrame,
                InferRequest& inferRequest,
                Context& context) {
                    inferRequest.SetCompletionCallback([] {});  // destroy the stored bind object
                    tryPush(context.detectionsProcessorsContext.detectionsProcessorsWorker,
                        std::make_shared<DetectionsProcessor>(sharedVideoFrame, &inferRequest));
            }, sharedVideoFrame,
            inferRequest,
                std::ref(context)));
    inferRequest.get().StartAsync();
    // do not push as callback does it
}

bool Reader::isReady() {
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    context.readersContext.lastCapturedFrameIdsMutexes[sharedVideoFrame->sourceID].lock();
    if (context.readersContext.lastCapturedFrameIds[sharedVideoFrame->sourceID] + 1 == sharedVideoFrame->frameId) {
        return true;
    }
    else {
        context.readersContext.lastCapturedFrameIdsMutexes[sharedVideoFrame->sourceID].unlock();
        return false;
    }
}

void Reader::process() {
    unsigned sourceID = sharedVideoFrame->sourceID;
    Context& context = static_cast<ReborningVideoFrame*>(sharedVideoFrame.get())->context;
    const std::vector<std::shared_ptr<InputChannel>>& inputChannels = context.readersContext.inputChannels;
    if (inputChannels[sourceID]->read(sharedVideoFrame->frame)) {
        context.readersContext.lastCapturedFrameIds[sourceID]++;
        context.readersContext.lastCapturedFrameIdsMutexes[sourceID].unlock();
        tryPush(context.inferTasksContext.inferTasksWorker, std::make_shared<InferTask>(sharedVideoFrame));
    }
    else {
        context.readersContext.lastCapturedFrameIds[sourceID]++;
        context.readersContext.lastCapturedFrameIdsMutexes[sourceID].unlock();
        try {
            std::shared_ptr<Worker>(context.drawersContext.drawersWorker)->stop();
        }
        catch (const std::bad_weak_ptr&) {}
    }
}

int main(int argc, char* argv[]) {
    try {

        slog::info << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << slog::endl;

        std::vector<std::shared_ptr<VideoCaptureSource>> videoCapturSourcess;
        std::vector<std::shared_ptr<ImageSource>> imageSourcess;

        cv::VideoCapture videoCapture(0); 
        if (!videoCapture.isOpened()) {
            slog::info << "Cannot open camera " << slog::endl;
            return 1;
        }
        videoCapturSourcess.push_back(std::make_shared<VideoCaptureSource>(videoCapture, FLAGS_loop_video));

        uint32_t channelsNum = videoCapturSourcess.size() + imageSourcess.size();
        std::cout << "videoCapturSourcess size, imageSourcess size " << videoCapturSourcess.size() << imageSourcess.size() << channelsNum << std::endl;
        std::cout << "channelsNum " << channelsNum << std::endl;

        std::vector<std::shared_ptr<IInputSource>> inputSources;
        inputSources.reserve(videoCapturSourcess.size() + imageSourcess.size());

        for (const std::shared_ptr<VideoCaptureSource>& videoSource : videoCapturSourcess) {
            inputSources.push_back(videoSource);
        }
        for (const std::shared_ptr<ImageSource>& imageSource : imageSourcess) {
            inputSources.push_back(imageSource);
        }

        std::vector<std::shared_ptr<InputChannel>> inputChannels;
        inputChannels.reserve(channelsNum);
        for (decltype(inputSources.size()) channelI = 0, counter = 0; counter < channelsNum; channelI++, counter++) {
            if (inputSources.size() == channelI) {
                channelI = 0;
            }
            inputChannels.push_back(InputChannel::create(inputSources[channelI]));
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Inference Engine -------------------------------------
        InferenceEngine::Core ie;

        std::set<std::string> devices;
        for (const std::string& netDevices : { vehicle_device, vehicle_Attribute_device, plate_Licese_device }) {
            if (netDevices.empty()) {
                continue;
            }
            for (const std::string& device : parseDevices(netDevices)) {
                devices.insert(device);
            }
        }
        std::map<std::string, uint32_t> device_nstreams;

        for (const std::string& device : devices) {
            slog::info << "Loading device " << device << slog::endl;

            /** Printing device version **/
            std::cout << ie.GetVersions(device) << std::endl;

            if ("CPU" == device) {
                if (FLAGS_nthreads != 0) {
                    ie.SetConfig({ { CONFIG_KEY(CPU_THREADS_NUM), std::to_string(FLAGS_nthreads) } }, "CPU");
                }
                ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) } }, "CPU");
                ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                    CONFIG_VALUE(CPU_THROUGHPUT_AUTO) } }, "CPU");
                device_nstreams["CPU"] = std::stoi(ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>());
            }

            if ("GPU" == device) {
                ie.SetConfig({ { CONFIG_KEY(GPU_THROUGHPUT_STREAMS),
                    CONFIG_VALUE(GPU_THROUGHPUT_AUTO) } }, "GPU");

                device_nstreams["GPU"] = std::stoi(ie.GetConfig("GPU", CONFIG_KEY(GPU_THROUGHPUT_STREAMS)).as<std::string>());
                
                if (devices.end() != devices.find("CPU")) {
                    ie.SetConfig({ { CLDNN_CONFIG_KEY(PLUGIN_THROTTLE), "1" } }, "GPU");
                }
            }
        }

        std::cout << "device_nstreams " << device_nstreams.size() << std::endl;

        /** Graph tagging via config options**/
        std::map<std::string, std::string> makeTagConfig;

        // -----------------------------------------------------------------------------------------------------
        std::cout << "inputChannels size " << inputChannels.size() << std::endl;
        slog::info << "Loading detection model to the " << vehicle_device << " plugin" << slog::endl;
        // FLAGS_m
        Detector detector(ie, vehicle_device, vehicleLicense_Model,
            { static_cast<float>(threshold_flag), static_cast<float>(threshold_flag) }, auto_Resize_FLAG, makeTagConfig);
        VehicleAttributesClassifier vehicleAttributesClassifier;
        std::size_t nclassifiersireq{ 0 };
        Lpr lpr;
        std::size_t nrecognizersireq{ 0 };

        slog::info << "Loading Vehicle Attribs model to the " << vehicle_Attribute_device << " plugin" << slog::endl;
        vehicleAttributesClassifier = VehicleAttributesClassifier(ie, vehicle_Attribute_device, vehicleAttribute_Model, auto_Resize_FLAG, makeTagConfig); //FLAGS_m_va FLAGS_auto_resize
        nclassifiersireq = inputChannels.size() * 3;
        
        slog::info << "Loading Licence Plate Recognition (LPR) model to the " << plate_Licese_device << " plugin" << slog::endl;
        lpr = Lpr(ie, plate_Licese_device, plateLicense_Model, auto_Resize_FLAG, makeTagConfig);
        nrecognizersireq = inputChannels.size() * 3;
        
        bool isVideo = imageSourcess.empty() ? true : false;
        int pause = imageSourcess.empty() ? 1 : 0;
        std::chrono::steady_clock::duration showPeriod = std::chrono::steady_clock::duration::zero();
        std::vector<cv::Size> gridParam;
        gridParam.reserve(inputChannels.size());
        for (const auto& inputChannel : inputChannels) {
            gridParam.emplace_back(inputChannel->getSize());
        }
        cv::Size displayResolution = cv::Size{ 1920, 1080 };

        slog::info << "Number of InferRequests: " << inputChannels.size() << " (detection), " << nclassifiersireq << " (classification), " << nrecognizersireq << " (recognition)" << slog::endl;
        std::ostringstream device_ss;
        for (const auto& nstreams : device_nstreams) {
            if (!device_ss.str().empty()) {
                device_ss << ", ";
            }
            device_ss << nstreams.second << " streams for " << nstreams.first;
        }
        if (!device_ss.str().empty()) {
            slog::info << device_ss.str() << slog::endl;
        }
        slog::info << "Display resolution: " << displayResolution << slog::endl;

        Context context{ inputChannels,
                        detector,
                        vehicleAttributesClassifier, lpr,
                        pause, gridParam, displayResolution, showPeriod, "",
                        numberOfallocatedFrame - 1,
                        inputChannels.size(),
                        isVideo,
                        nclassifiersireq, nrecognizersireq };

        std::shared_ptr<Worker> worker = std::make_shared<Worker>(nunber_workerThread - 1);
        context.readersContext.readersWorker = context.inferTasksContext.inferTasksWorker
            = context.detectionsProcessorsContext.detectionsProcessorsWorker = context.drawersContext.drawersWorker
            = context.resAggregatorsWorker = worker;

        for (uint64_t i = 0; i < numberOfallocatedFrame; i++) {
            for (unsigned sourceID = 0; sourceID < inputChannels.size(); sourceID++) {
                VideoFrame::Ptr sharedVideoFrame = std::make_shared<ReborningVideoFrame>(context, sourceID, i);
                worker->push(std::make_shared<Reader>(sharedVideoFrame));
            }
        }
        slog::info << "Number of allocated frames: " << numberOfallocatedFrame * (inputChannels.size()) << slog::endl;
        if (auto_Resize_FLAG) { 
            slog::info << "Resizable input with support of ROI crop and auto resize is enabled" << slog::endl;
        }
        else {
            slog::info << "Resizable input with support of ROI crop and auto resize is disabled" << slog::endl;
        }

        // Running
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        context.t0 = t0;
        context.drawersContext.updateTime = t0;
        worker->runThreads();
        worker->threadFunc();
        worker->join();
        const auto t1 = std::chrono::steady_clock::now();

        std::map<std::string, std::string> mapDevices = getMapFullDevicesNames(ie, { vehicle_device, vehicle_Attribute_device, plate_Licese_device });
        for (auto& net : std::array<std::pair<std::vector<InferRequest>, std::string>, 3>{
            std::make_pair(context.detectorsInfers.getActualInferRequests(), vehicle_device),
                std::make_pair(context.attributesInfers.getActualInferRequests(), vehicle_Attribute_device),
                std::make_pair(context.platesInfers.getActualInferRequests(), plate_Licese_device)}) {
            for (InferRequest& ir : net.first) {
                ir.Wait(IInferRequest::WaitMode::RESULT_READY);
            }
        }

        uint64_t frameCounter = context.frameCounter;
        if (0 != frameCounter) {
            const float fps = static_cast<float>(frameCounter) / std::chrono::duration_cast<Sec>(t1 - context.t0).count()
                / context.readersContext.inputChannels.size();
            std::cout << std::fixed << std::setprecision(1) << fps << "FPS for (" << frameCounter << " / "
                << inputChannels.size() << ") frames\n";
            const double detectionsInfersUsage = static_cast<float>(frameCounter * context.nireq - context.freeDetectionInfersCount)
                / (frameCounter * context.nireq) * 100;
            std::cout << "Detection InferRequests usage: " << detectionsInfersUsage << "%\n";
        }

        std::cout << context.drawersContext.presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;
    return 0;
}
