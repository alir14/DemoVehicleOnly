#include "ObjBlob.hpp"

ObjBlob::ObjBlob(Detector::Result detectedObj, std::string pln, int index, cv::Mat& frame)
{
	frameIndex = index;
	plateNumber = pln;
	detectedObject = detectedObj;
	_boundingRect = detectedObj.location;

	cv::Point currentCenter = CalculateCenterPoint(_boundingRect);

	centerPositions.push_back(currentCenter);

	dblDiagonalSize = sqrt(pow(_boundingRect.width, 2) + pow(_boundingRect.height, 2));
	dblAspectRatio = (float)_boundingRect.width / (float)_boundingRect.height;

	blnStillBeingTracked = true;
	blnCurrentMatchFoundOrNewBlob = true;
}

void ObjBlob::PredictNextPosition()
{
	int numPositions = (int)centerPositions.size();

	if (numPositions == 1) {

		predictedNextPosition.x = centerPositions.back().x;
		predictedNextPosition.y = centerPositions.back().y;

	}
	else if (numPositions == 2) {

		int deltaX = centerPositions[1].x - centerPositions[0].x;
		int deltaY = centerPositions[1].y - centerPositions[0].y;

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions == 3) {

		int sumOfXChanges = ((centerPositions[2].x - centerPositions[1].x) * 2) +
			((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 3.0);

		int sumOfYChanges = ((centerPositions[2].y - centerPositions[1].y) * 2) +
			((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 3.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions == 4) {

		int sumOfXChanges = ((centerPositions[3].x - centerPositions[2].x) * 3) +
			((centerPositions[2].x - centerPositions[1].x) * 2) +
			((centerPositions[1].x - centerPositions[0].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 6.0);

		int sumOfYChanges = ((centerPositions[3].y - centerPositions[2].y) * 3) +
			((centerPositions[2].y - centerPositions[1].y) * 2) +
			((centerPositions[1].y - centerPositions[0].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 6.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
	else if (numPositions >= 5) {

		int sumOfXChanges = ((centerPositions[numPositions - 1].x - centerPositions[numPositions - 2].x) * 4) +
			((centerPositions[numPositions - 2].x - centerPositions[numPositions - 3].x) * 3) +
			((centerPositions[numPositions - 3].x - centerPositions[numPositions - 4].x) * 2) +
			((centerPositions[numPositions - 4].x - centerPositions[numPositions - 5].x) * 1);

		int deltaX = (int)std::round((float)sumOfXChanges / 10.0);

		int sumOfYChanges = ((centerPositions[numPositions - 1].y - centerPositions[numPositions - 2].y) * 4) +
			((centerPositions[numPositions - 2].y - centerPositions[numPositions - 3].y) * 3) +
			((centerPositions[numPositions - 3].y - centerPositions[numPositions - 4].y) * 2) +
			((centerPositions[numPositions - 4].y - centerPositions[numPositions - 5].y) * 1);

		int deltaY = (int)std::round((float)sumOfYChanges / 10.0);

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;

	}
}

int ObjBlob::CalculateDeltaX(std::vector<cv::Point>& positions)
{
	const int limit = 5;
	int numberOfPos = positions.size();
	int sum = 0;
	int divideValue = 0;
	for (size_t i = 1; i < limit; i++)
	{
		int multiplValue = (numberOfPos <= 5) ? (numberOfPos - i) : (5 - i);

		divideValue = divideValue + multiplValue;

		if ((numberOfPos - (i + 1)) >= 0 && (numberOfPos - i) > 0)
		{
			sum += ((positions[numberOfPos - i].x - positions[numberOfPos - (i + 1)].x) * multiplValue);
		}
		else
			break;
	}

	return (int)std::round((float)sum / (float)divideValue);
}

int ObjBlob::CalculateDeltaY(std::vector<cv::Point>& positions)
{
	const int limit = 5;
	int numberOfPos = positions.size();
	int sum = 0;
	int divideValue = 0;
	for (size_t i = 1; i < limit; i++)
	{
		int multiplValue = (numberOfPos <= 5) ? (numberOfPos - i) : (5 - i);

		divideValue = divideValue + multiplValue;

		if ((numberOfPos - (i + 1)) >= 0 && (numberOfPos - i) > 0)
		{
			sum += ((positions[numberOfPos - i].y - positions[numberOfPos - (i + 1)].y) * multiplValue);
		}
		else
			break;
	}

	return  (int)std::round((float)sum / (float)divideValue);
}

cv::Point ObjBlob::CalculateCarPositoion(cv::Mat& currentFrame, cv::Mat& prevFrame)
{
	cv::Mat imgDifference, imgThresh;
	cv::Mat current = currentFrame.clone();
	cv::Mat previous = prevFrame.clone();

	cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
	cv::cvtColor(previous, previous, cv::COLOR_BGR2GRAY);

	cv::GaussianBlur(current, current, cv::Size(5, 5), 0);
	cv::GaussianBlur(previous, previous, cv::Size(5, 5), 0);

	cv::absdiff(previous, current, imgDifference);

	cv::threshold(imgDifference, imgThresh, 30, 255.0, cv::THRESH_BINARY);

	cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	for (unsigned int i = 0; i < 2; i++) {
		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		cv::dilate(imgThresh, imgThresh, structuringElement5x5);
		cv::erode(imgThresh, imgThresh, structuringElement5x5);
	}

	cv::Mat imgThreshCopy = imgThresh.clone();

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(imgThreshCopy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	int idx = 0, largestComp = 0;
	double maxArea = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		const std::vector<cv::Point>& c = contours[idx];
		double area = fabs(cv::contourArea(cv::Mat(c)));
		if (area > maxArea)
		{
			maxArea = area;
			largestComp = idx;
		}
	}

	cv::Rect boundingContour = cv::boundingRect(contours[largestComp]);
	cv::Point lastCenterPoint = centerPositions.back();

	cv::Mat image(current.size(), CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

	cv::rectangle(image, boundingContour, cv::Scalar(0.0, 0.0, 255.0), 4);
	cv::circle(image, lastCenterPoint, 10, cv::Scalar(0.0, 255.0, 255.0), -1); //yellow
	PredictNextPosition();
	cv::circle(image, predictedNextPosition, 10, cv::Scalar(255.0, 255.0, 255.0), -1); // white
	if (lastCenterPoint.inside(boundingContour))
	{
		cv::Point result = CalculateCenterPoint(boundingContour);
		cv::circle(image, result, 10, cv::Scalar(0.0, 200.0, 0.0), -1);  //green

		cv::imshow("debug", image);

		return result;
	}

	return cv::Point(0, 0);
}

cv::Point ObjBlob::CalculateCenterPoint(cv::Rect rect)
{
	cv::Point point;

	point.x = ((rect.x * 2) + rect.width) / 2;
	point.y = ((rect.y * 2) + rect.height) / 2;
	
	return point;
}

void ObjBlob::drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) 
{
	cv::Mat image(imageSize, CV_8UC3, cv::Scalar(0.0, 0.0, 0.0));

	cv::drawContours(image, contours, -1, cv::Scalar(255.0, 255.0, 255.0), -1);

	cv::imshow(strImageName, image);
}
