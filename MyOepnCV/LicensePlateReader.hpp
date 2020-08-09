#pragma once
#include <list>
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>

class LicensePlateReader
{
public:
	LicensePlateReader() = default;
	LicensePlateReader(
        InferenceEngine::Core& ie, 
		const std::string& xmlPath, 
		const std::map<std::string, std::string>& pluginConfig);

    InferenceEngine::InferRequest createInferRequest();
    std::string ProcessAndReadPalteNumber(
        InferenceEngine::InferRequest& inferRequest, 
        const cv::Mat& img,
        const cv::Rect plateRect);

private:
    void setImage(
        InferenceEngine::InferRequest& inferRequest, 
        const cv::Mat& img, 
        const cv::Rect plateRect);
    std::string getResult(InferenceEngine::InferRequest& inferRequest);

    int maxSequenceSizePerPlate;
    std::string LprInputName;
    std::string LprInputSeqName;
    std::string LprOutputName;
    InferenceEngine::Core ie_;  // The only reason to store a device as to assure that it lives at least as long as ExecutableNetwork
    InferenceEngine::ExecutableNetwork net;
};

