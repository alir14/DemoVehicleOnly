#pragma once
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

#include <opencv2/core.hpp>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>

class Detector
{
    static constexpr int maxProposalCount = 200;
    static constexpr int objectSize = 7;  // Output should have 7 as a last dimension"

public:
    struct Result {
        std::size_t label;
        float confidence;
        cv::Rect location;
    };

    Detector() = default;
    Detector(InferenceEngine::Core& ie,
        const std::string& xmlPath,
        const std::vector<float>& detectionTresholds,
        const std::map<std::string, std::string>& pluginConfig);

    InferenceEngine::InferRequest createInferRequest();
    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img);
    std::list<Result> getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale);

private:
    std::vector<float> detectionTresholds;
    std::string detectorInputBlobName;
    std::string detectorOutputBlobName;
    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
    void Logging(const char* msg);
};

