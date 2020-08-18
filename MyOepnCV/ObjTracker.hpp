#pragma once

#include <vector>
#include "ObjBlob.hpp"

class ObjTracker
{
public:
	std::vector<ObjBlob> blobs;
	std::vector<ObjBlob> currentFrameBlobs;
	void matchCurrentFrameBlobsToExistingBlobs(int frameIndex);
	
	void addNewBlob(ObjBlob& currentFrameBlob);
	double distanceBetweenPoints(cv::Point point1, cv::Point point2);
	void drawBlobInfoOnImage(std::vector<ObjBlob>& blobs, cv::Mat& img);
	void TrackMissedObject(ObjBlob& missedBlob, cv::Mat& currentFrame, cv::Mat& prevFrame, int frameIndex);

private:
	void addBlobToExistingBlobs(ObjBlob& currentFrameBlob, int& intIndex, int frameIndex);
	const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
	const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
};

