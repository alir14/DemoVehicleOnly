#include "Detector.hpp"

Detector::Detector(InferenceEngine::Core& ie,
    const std::string& xmlPath,
    const std::vector<float>& detectionTresholds,
    const std::map<std::string, std::string>& pluginConfig) :
    detectionTresholds{ detectionTresholds }, ie_{ ie }
{
    auto network = ie.ReadNetwork(xmlPath);

    InferenceEngine::InputsDataMap inputInfo(network.getInputsInfo());
    if (inputInfo.size() != 1)
    {
        throw std::logic_error("Detector should have only one input");
    }

    InferenceEngine::InputInfo::Ptr& inputInfoFirst = inputInfo.begin()->second;
    inputInfoFirst->setPrecision(InferenceEngine::Precision::U8);

    inputInfoFirst->getPreProcess().setResizeAlgorithm(InferenceEngine::ResizeAlgorithm::RESIZE_BILINEAR);
    inputInfoFirst->setLayout(InferenceEngine::Layout::NHWC);

    detectorInputBlobName = inputInfo.begin()->first;

    InferenceEngine::OutputsDataMap outputInfo(network.getOutputsInfo());

    if (outputInfo.size() != 1)
    {
        throw std::logic_error("Vehicle Detection network should have only one output");
    }

    InferenceEngine::DataPtr& _output = outputInfo.begin()->second;

    const InferenceEngine::SizeVector outputDims = _output->getTensorDesc().getDims();

    detectorOutputBlobName = outputInfo.begin()->first;

    if (maxProposalCount != outputDims[2])
    {
        throw std::logic_error("unexpected ProposalCount");
    }
    if (objectSize != outputDims[3])
    {
        throw std::logic_error("Output should have 7 as a last dimension");
    }
    if (outputDims.size() != 4)
    {
        throw std::logic_error("Incorrect output dimensions for SSD");
    }

    _output->setPrecision(InferenceEngine::Precision::FP32);

    net = ie_.LoadNetwork(network, "CPU", pluginConfig);
}

InferenceEngine::InferRequest Detector::createInferRequest()
{
    return net.CreateInferRequest();
}

void Detector::setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img)
{
    InferenceEngine::Blob::Ptr input = inferRequest.GetBlob(detectorInputBlobName);
    if (InferenceEngine::Layout::NHWC == input->getTensorDesc().getLayout())
    {
        if (!img.isSubmatrix())
        {
            InferenceEngine::Blob::Ptr frameBlob = wrapMat2Blob(img);
            inferRequest.SetBlob(detectorInputBlobName, frameBlob);
        }
        else
        {
            throw std::logic_error("Sparse matrix are not supported");
        }
    }
    else
    {
        matU8ToBlob<uint8_t>(img, input);
    }
}

std::list<Detector::Result> Detector::getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale)
{
    std::list<Detector::Result> results;

    const float* const detections = inferRequest.GetBlob(detectorOutputBlobName)->buffer().as<float*>();

    for (size_t i = 0; i < maxProposalCount; i++)
    {
        float image_id = detections[i * objectSize + 0];  // in case of batch
        
        if (image_id < 0) {  // indicates end of detections
            break;
        }

        auto label = static_cast<decltype(detectionTresholds.size())>(detections[i * objectSize + 1]);
        
        float confidence = detections[i * objectSize + 2];
        if (label - 1 < detectionTresholds.size() && confidence < detectionTresholds[label - 1]) {
            continue;
        }

        cv::Rect rect;
        rect.x = static_cast<int>(detections[i * objectSize + 3] * upscale.width);
        rect.y = static_cast<int>(detections[i * objectSize + 4] * upscale.height);
        rect.width = static_cast<int>(detections[i * objectSize + 5] * upscale.width) - rect.x;
        rect.height = static_cast<int>(detections[i * objectSize + 6] * upscale.height) - rect.y;

        results.push_back(Result{ label, confidence, rect });

        std::cout << "[" << i << "," << label << "] element, prob = " << confidence
            << "    (" << rect.x << "," << rect.y << ")-(" << rect.width << "," << rect.height << ")" << std::endl;
    }
    return results;
}

std::list<Detector::Result> Detector::ProcessAndReturnResult(InferenceEngine::InferRequest& inferenceReq, const cv::Mat& img)
{
    cv::Mat currentFrame = img.clone();

    setImage(inferenceReq, currentFrame);

    inferenceReq.Infer();

    return getResults(inferenceReq, img.size());
}