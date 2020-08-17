#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include <cldnn/cldnn_config.hpp>
#include <inference_engine.hpp>

#include "Detector.hpp"
#include "LicensePlateReader.hpp"
#include "BlobObj.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<BlobObj>& existingBlobs, std::vector<BlobObj>& currentFrameBlobs);
void addBlobToExistingBlobs(BlobObj& currentFrameBlob, std::vector<BlobObj>& existingBlobs, int& intIndex);
void addNewBlob(BlobObj& currentFrameBlob, std::vector<BlobObj>& existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<BlobObj> blobs, std::string strImageName);

using namespace InferenceEngine;

int main()
{
	std::cout << "InferenceEngine: " << InferenceEngine::GetInferenceEngineVersion() << std::endl;

	std::map<std::string, std::string> pluginConfig;
	std::vector<float> thresholdVector{ static_cast<float>(0.5), static_cast<float>(0.5) };
	std::string vehicleLicense_Model = "D:\\workspace\\openvino\\models\\intel\\vehicle-license-plate-detection-barrier-0106\\FP32\\vehicle-license-plate-detection-barrier-0106.xml";
	std::string plateLicense_Model = "D:\\workspace\\openvino\\models\\intel\\license-plate-recognition-barrier-0001\\FP32\\license-plate-recognition-barrier-0001.xml";
	
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
	cv::Mat frame;
	std::string path = "D:\\media\\cars2.mp4";
	cv::VideoCapture cap;


    cv::Mat imgFrame1, imgFrame2;
    std::vector<BlobObj> blobs;
    cv::Point crossingLine[2];
    int carCount = 0;

	if (!cap.open(path)) 
	{
		std::cerr << "cannot open the media" << std::endl;
		return 0;
	}

	cv::Rect rectROI;


    cap.read(imgFrame1);
    cap.read(imgFrame2);

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.35);

    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;
    cv::Mat imgFrame1Copy, imgFrame2Copy, imgDifference, imgThresh;

    std::vector<BlobObj> currentFrameBlobs;
	while (cap.isOpened() && chCheckForEscKey != 27)
	{
		cap >> frame;

		if (frame.empty())
			break;

        imgFrame1Copy = imgFrame1.clone();
        imgFrame2Copy = imgFrame2.clone();

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30, 255.0, cv::THRESH_BINARY);

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        //drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        //drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        for (auto& convexHull : convexHulls) {
            BlobObj possibleBlob(convexHull);

            if (
                possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 70 &&
                possibleBlob.currentBoundingRect.height > 70 &&
                possibleBlob.dblCurrentDiagonalSize > 70.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.70) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

        if (blnFirstFrame == true) {
            for (auto& currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        }
        else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }

        std::cout << "blobs size " << blobs.size() << std::endl;

        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        cv::imshow("imgFrame2Copy", imgFrame2Copy);

        // now we prepare for the next iteration
        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((cap.get(cv::CAP_PROP_POS_FRAMES) + 1) < cap.get(cv::CAP_PROP_FRAME_COUNT)) {
            cap.read(imgFrame2);
        }
        else {
            std::cout << "end of video\n";
            break;
        }

        currentFrameBlobs.clear();
        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);

		//process
		//auto results = detect.ProcessAndReturnResult(detectInfReq, frame);

		//for (Detector::Result result : results)
		//{
		//	if (result.label == 1)
		//	{
		//		rectROI = cv::Rect(result.location.x, result.location.y, result.location.width, result.location.height);
		//		cv::rectangle(frame, rectROI, { 255,255,0 }, 2);
		//	}
		//	else if (result.label == 2)
		//	{
		//		rectROI = cv::Rect(abs(result.location.x), abs(result.location.y), result.location.width + 5, result.location.height + 5);
		//		cv::rectangle(frame, rectROI, { 255,255,100 }, 2);

		//		if (result.confidence > 0.5)
		//		{
		//			std::string paltenumber = lpr.ProcessAndReadPalteNumber(lprInferReq, frame, rectROI);

		//			if (!paltenumber.empty())
		//			{
		//				cv::putText(frame, paltenumber, cv::Point(abs(result.location.x), abs(result.location.y - 5)), 
		//					cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
		//			}
		//		}
		//	}
		//}

		//cv::imshow("frame", frame);

		//if (cv::waitKey(5) >= 0) break;
	}

	cv::waitKey(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<BlobObj>& existingBlobs, std::vector<BlobObj>& currentFrameBlobs) {

    for (auto& existingBlob : existingBlobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.predictNextPosition();
    }

    for (auto& currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto& existingBlob : existingBlobs) {

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(BlobObj& currentFrameBlob, std::vector<BlobObj>& existingBlobs, int& intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(BlobObj& currentFrameBlob, std::vector<BlobObj>& existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<BlobObj> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto& blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

