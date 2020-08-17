#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Detector.hpp"
#include <vector>

class ObjBlob
{
public:
	int frameIndex;
	std::string plateNumber;
	cv::Point predictedNextPosition;
	bool blnStillBeingTracked;
	cv::Rect _boundingRect;
	std::vector<cv::Point> centerPositions;
	double dblDiagonalSize = 0;
	double dblAspectRatio = 0;
	bool blnCurrentMatchFoundOrNewBlob;
	Detector::Result detectedObject;
	int intNumOfConsecutiveFramesWithoutAMatch = 0;

	ObjBlob(Detector::Result detectedObj, std::string pln, int index);
	void PredictNextPosition();
	
private:
	int CalculateDeltaX(std::vector<cv::Point>& positions);
	int CalculateDeltaY(std::vector<cv::Point>& positions);
	
};

