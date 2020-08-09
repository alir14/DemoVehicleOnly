#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include <cldnn/cldnn_config.hpp>
#include <inference_engine.hpp>

#include "Detector.hpp"
#include "LicensePlateReader.hpp"


using namespace InferenceEngine;

int main()
{
	std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

	std::map<std::string, std::string> pluginConfig;
	std::vector<float> thresholdVector{ static_cast<float>(0.5), static_cast<float>(0.5) };
	std::string vehicleLicense_Model = "D:\\workspace\\openvino\\models\\intel\\vehicle-license-plate-detection-barrier-0106\\FP16\\vehicle-license-plate-detection-barrier-0106.xml";
	std::string plateLicense_Model = "D:\\workspace\\openvino\\models\\intel\\license-plate-recognition-barrier-0001\\FP16\\license-plate-recognition-barrier-0001.xml";
	
	InferenceEngine::Core ie;

	std::cout << ie.GetVersions("CPU") << std::endl;

	ie.SetConfig({ { CONFIG_KEY(CPU_BIND_THREAD), CONFIG_VALUE(NO) } }, "CPU");
	ie.SetConfig({ { CONFIG_KEY(CPU_THROUGHPUT_STREAMS), CONFIG_VALUE(CPU_THROUGHPUT_AUTO) } }, "CPU");

	std::string numOfStreamValue = ie.GetConfig("CPU", CONFIG_KEY(CPU_THROUGHPUT_STREAMS)).as<std::string>();
	uint32_t numOfCPUStream = std::stoi(numOfStreamValue);
	std::cout << "number of CPU stream " << numOfCPUStream << std::endl;

	Detector detect(ie, vehicleLicense_Model, thresholdVector, pluginConfig);
	LicensePlateReader lpr(ie, plateLicense_Model, pluginConfig);

	InferenceEngine::InferRequest detectInfReq = detect.createInferRequest();
	InferenceEngine::InferRequest lprInferReq = lpr.createInferRequest();

	//cv::Mat frame = cv::imread("D:\\media\\redcar1.jpg");
	cv::Mat frame;
	std::string path = "D:\\media\\sample.mp4";
	cv::VideoCapture cap;

	if (!cap.open(path)) 
	{
		std::cerr << "cannot open the media" << std::endl;
		return 0;
	}

	while (true)
	{
		cap >> frame;

		if (frame.empty())
			break;

		//process
		auto results = detect.ProcessAndReturnResult(detectInfReq, frame);

		for (Detector::Result result : results)
		{
			switch (result.label)
			{
			case 1:
			{
				cv::Rect rect = cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height);
				cv::rectangle(frame, rect, { 255,255,0 }, 2);
				break;
			}
			case 2:
			{
				std::string paltenumber = lpr.ProcessAndReadPalteNumber(lprInferReq, frame, result.location);
				if (!paltenumber.empty())
					cv::putText(frame, paltenumber, cv::Point(20, 50), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
				break;
			}
			default:
				break;
			}
		}

		cv::imshow("frame", frame);

		if (cv::waitKey(5) >= 0) break;
	}

	cv::waitKey(0);
}