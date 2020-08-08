#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include <cldnn/cldnn_config.hpp>
#include <inference_engine.hpp>

#include "Detector.hpp"

using namespace InferenceEngine;

int main()
{
	std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

	std::map<std::string, std::string> pluginConfig;
	std::vector<float> thresholdVector{ static_cast<float>(0.5), static_cast<float>(0.5) };
	std::string vehicleLicense_Model = "D:\\workspace\\openvino\\models\\intel\\vehicle-license-plate-detection-barrier-0106\\FP16\\vehicle-license-plate-detection-barrier-0106.xml";

	InferenceEngine::Core ie;

	std::cout << ie.GetVersions("CPU") << std::endl;

	ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) } }, "CPU");
	ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS), CONFIG_VALUE(CPU_THROUGHPUT_AUTO) } }, "CPU");

	std::string numOfStreamValue = ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>();
	uint32_t numOfCPUStream = std::stoi(numOfStreamValue);
	std::cout << "number of CPU stream " << numOfCPUStream << std::endl;

	Detector detect(ie, vehicleLicense_Model, thresholdVector, pluginConfig);

	InferenceEngine::InferRequest infReq = detect.createInferRequest();

	cv::Mat frame = cv::imread("D:\\media\\redcar1.jpg");

	detect.setImage(infReq, frame);

	infReq.Infer();

	auto results = detect.getResults(infReq, frame.size());

	for (Detector::Result result : results)
	{
		cv::Rect rect = cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height);
		cv::rectangle(frame, rect, {255,255,0}, 2);
	}

	cv::imshow("frame", frame);

	cv::waitKey(0);
}