#pragma once
#include <list>
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <inference_engine.hpp>
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

    std::list<Result> ProcessAndReturnResult(InferenceEngine::InferRequest& inferenceReq, const cv::Mat& img);
    InferenceEngine::InferRequest createInferRequest();

private:
    std::list<Result> getResults(InferenceEngine::InferRequest& inferRequest, cv::Size upscale);
    void setImage(InferenceEngine::InferRequest& inferRequest, const cv::Mat& img);
    std::vector<float> detectionTresholds;
    std::string detectorInputBlobName;
    std::string detectorOutputBlobName;
    InferenceEngine::Core ie_;  // The only reason to store a plugin as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};

