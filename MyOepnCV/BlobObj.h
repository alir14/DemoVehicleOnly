// Blob.h
#pragma once
#ifndef MY_BLOB
#define MY_BLOB

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

///////////////////////////////////////////////////////////////////////////////////////////////////
class BlobObj {
public:
    // member variables ///////////////////////////////////////////////////////////////////////////
    std::vector<cv::Point> currentContour;

    cv::Rect currentBoundingRect;

    std::vector<cv::Point> centerPositions;

    double dblCurrentDiagonalSize;
    double dblCurrentAspectRatio;

    bool blnCurrentMatchFoundOrNewBlob;

    bool blnStillBeingTracked;

    int intNumOfConsecutiveFramesWithoutAMatch;

    cv::Point predictedNextPosition;

    // function prototypes ////////////////////////////////////////////////////////////////////////
    BlobObj(std::vector<cv::Point> _contour);
    void predictNextPosition(void);

};

#endif    // MY_BLOB

