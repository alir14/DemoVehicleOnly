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
	int deltaX, deltaY = 0;

	if (numPositions == 1) {

		predictedNextPosition.x = centerPositions.back().x;
		predictedNextPosition.y = centerPositions.back().y;

	}
	else if (numPositions == 2) {

		deltaX = centerPositions[1].x - centerPositions[0].x;
		deltaY = centerPositions[1].y - centerPositions[0].y;

		predictedNextPosition.x = centerPositions.back().x + deltaX;
		predictedNextPosition.y = centerPositions.back().y + deltaY;
	}
	else if (numPositions >= 3) {
		deltaX = CalculateDeltaX(centerPositions);
		deltaY = CalculateDeltaY(centerPositions);

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

	cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point> > convexHulls(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++) {
		cv::convexHull(contours[i], convexHulls[i]);
	}
	drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

	for (auto& convexHull : convexHulls) {
		cv::Rect boundingContour = cv::boundingRect(convexHull);
		if (boundingContour.area() > 400 &&
			dblAspectRatio > 0.2 &&
			dblAspectRatio < 4.0 &&
			boundingContour.width > 50 &&
			boundingContour.height > 50 &&
			dblDiagonalSize > 60.0 &&
			(cv::contourArea(convexHull) / (double)boundingContour.area()) > 0.7)
		{
			cv::Point lastCenterPoint = centerPositions.back();
			if (lastCenterPoint.inside(boundingContour))
			{
				return CalculateCenterPoint(boundingContour);
			}
		}
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
