#include "ObjBlob.hpp"

ObjBlob::ObjBlob(Detector::Result detectedObj, std::string pln, int index)
{
	frameIndex = index;
	plateNumber = pln;
	detectedObject = detectedObj;
	_boundingRect = detectedObj.location;

	cv::Point currentCenter;

	currentCenter.x = ((_boundingRect.x * 2) + _boundingRect.width) / 2;
	currentCenter.y = ((_boundingRect.y * 2) + _boundingRect.height) / 2;

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