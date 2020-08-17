#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include <cldnn/cldnn_config.hpp>
#include <inference_engine.hpp>

#include "Detector.hpp"
#include "LicensePlateReader.hpp"

#include "ObjBlob.hpp"
#include "ObjTracker.hpp"

using namespace InferenceEngine;

int main()
{
	const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

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

	//cv::Mat frame = cv::imread("D:\\media\\test2.jpg");
	cv::Mat frame, prevFrame;
	std::string path = "D:\\media\\cars2.MP4";
	cv::VideoCapture cap;

	if (!cap.open(path)) 
	{
		std::cerr << "cannot open the media" << std::endl;
		return 0;
	}

	cv::Rect rectROI;
	ObjTracker carTracker;
	cv::Mat carImage;
	bool blnFirstFrame = true;
	int frameIndex = 0;
	while (true)
	{
		cap >> frame;

		if (frame.empty())
			break;

		//process
		auto results = detect.ProcessAndReturnResult(detectInfReq, frame);

		for (Detector::Result result : results)
		{
			if (result.label == 1)
			{
				rectROI = cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height);
				carImage = frame(rectROI);

				cv::rectangle(frame, rectROI, { 255,255,0 }, 2);
			}
			else if (result.label == 2)
			{
				std::cout << "detect and item ... " << std::endl;
				rectROI = cv::Rect(abs(result.location.x), abs(result.location.y), result.location.width + 5, result.location.height + 5);

				std::string paltenumber = lpr.ProcessAndReadPalteNumber(lprInferReq, frame, rectROI);

				ObjBlob plnBlob(result, paltenumber, frameIndex);
				carTracker.currentFrameBlobs.push_back(plnBlob);
		
				if (blnFirstFrame == true)
				{
					carTracker.addNewBlob(plnBlob);
				}
				else
				{
					carTracker.matchCurrentFrameBlobsToExistingBlobs(frameIndex);
				}
			}
		}

		std::cout << "number of blobs : " << carTracker.blobs.size() << std::endl;

		//std::cout << "number of current frame blobs : " << carTracker.currentFrameBlobs.size() << std::endl;

		for (auto& blobItem: carTracker.blobs)
		{
			if (blobItem.blnStillBeingTracked == true) {
				if (blobItem.frameIndex != frameIndex)
				{
					std::cout << "sync tracking " << blobItem.frameIndex << " - current " << frameIndex << std::endl;
					carTracker.TrackMissedObject(blobItem, frame.cols, frame.rows, frameIndex);
				}

				cv::circle(frame, blobItem.centerPositions.back(), 5, SCALAR_RED, -1);
				cv::circle(frame, blobItem.predictedNextPosition, 5, SCALAR_GREEN, -1);
				//std::cout << blobItem.predictedNextPosition.x << " - " << blobItem.predictedNextPosition.y << std::endl;

				if (!blobItem.plateNumber.empty())
				{
					cv::putText(frame, blobItem.plateNumber, cv::Point(abs(blobItem._boundingRect.x), abs(blobItem._boundingRect.y - 5)),cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
				}
			}
		}

		blnFirstFrame = false;
		cv::imshow("frame", frame);
		carTracker.currentFrameBlobs.clear();
		frameIndex++;
		if (cv::waitKey(5) >= 0) break;
	}

	cv::waitKey(0);
}
