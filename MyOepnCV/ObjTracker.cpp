#include "ObjTracker.hpp"

void ObjTracker::matchCurrentFrameBlobsToExistingBlobs(int frameIndex) { 

    for (auto& existingBlob : blobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.PredictNextPosition();
    }

    for (auto& currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < blobs.size(); i++) {

            if (blobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), blobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, intIndexOfLeastDistance, frameIndex);
        }
        else {
            addNewBlob(currentFrameBlob);
        }

    }

    for (auto& existingBlob : blobs) {
        std::cout << "checking if tracker is live " << std::endl;
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            std::cout << "tracker is out " << existingBlob.plateNumber << std::endl;
            existingBlob.blnStillBeingTracked = false;
        }
    }
}

void ObjTracker::addBlobToExistingBlobs(ObjBlob& currentFrameBlob, int& intIndex, int frameIndex) {
    blobs[intIndex].frameIndex = frameIndex;

    if (currentFrameBlob.detectedObject.confidence > blobs[intIndex].detectedObject.confidence) {
        blobs[intIndex].detectedObject = currentFrameBlob.detectedObject;
    }

    blobs[intIndex]._boundingRect = currentFrameBlob._boundingRect;

    blobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    blobs[intIndex].dblDiagonalSize = currentFrameBlob.dblDiagonalSize;
    blobs[intIndex].dblAspectRatio = currentFrameBlob.dblAspectRatio;

    blobs[intIndex].blnStillBeingTracked = true;
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

void ObjTracker::TrackMissedObject(ObjBlob& missedBlob, int width, int height, int frameIndex)
{
    std::cout << "found a missed item ... " << missedBlob.plateNumber << std::endl;
    std::cout << "last location ... " << missedBlob.centerPositions.back().x << " - " << missedBlob.centerPositions.back().y << std::endl;
    missedBlob.PredictNextPosition();
    std::cout << "predicted point is ... " << missedBlob.predictedNextPosition.x << " - " << missedBlob.predictedNextPosition.y << std::endl;

    if (missedBlob.predictedNextPosition.x < width && missedBlob.predictedNextPosition.y < height)
    {
        //missedBlob.frameIndex = frameIndex;
        missedBlob.detectedObject.location.x = missedBlob.predictedNextPosition.x;
        missedBlob.detectedObject.location.y = missedBlob.predictedNextPosition.y;

        missedBlob._boundingRect.x = missedBlob.predictedNextPosition.x;
        missedBlob._boundingRect.y = missedBlob.predictedNextPosition.y;

        missedBlob.centerPositions.push_back(missedBlob.predictedNextPosition);

        missedBlob.blnStillBeingTracked = true;
        std::cout << "after update last location is ... " << missedBlob.centerPositions.back().x << " - " << missedBlob.centerPositions.back().y << std::endl;
    }
    else
    {
        missedBlob.blnStillBeingTracked = false;
    }
}

void ObjTracker::addNewBlob(ObjBlob& currentFrameBlob) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    blobs.push_back(currentFrameBlob);
}

double ObjTracker::distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

void ObjTracker::drawBlobInfoOnImage(std::vector<ObjBlob>& blobs, cv::Mat& imgFrame2Copy) {

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnStillBeingTracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i]._boundingRect, SCALAR_RED, 2);

            int intFontFace = cv::FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        }
    }
}

