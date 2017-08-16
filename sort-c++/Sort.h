#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "KalmanTracker.h"

/** simple tracking bounding box
*/
struct TrackingBox
{
	/// frame number
	int frame;
	/// unique ID
	int id;
	/// the boundig box in top left coordinates, x,y,w,h
	cv::Rect_<float> box;
};

/** Sort tracking algorithm
*/
class Sort
{
private:
	/// max amount of time to keep an old tracking object around
	float max_age;
	/// the min amount of hits
	int min_hits;
	/// A vector of all current trackers
	std::vector<KalmanTracker*> trackers;
	/// tracks the frame count relative to the start of tracking
	int frame_count;
	/// The last tracking results
	std::vector<TrackingBox> results;
public:
	Sort();
	virtual ~Sort();
	/** update with latest detetions 
	 @param det A vector of opencv Rect objects
	 @return An updated list of tracking objects
	*/
	const std::vector<TrackingBox>& update(std::vector<cv::Rect>& det);
	/** resets the tracker
	*/
	void reset();

private:

	/** Assigns detections to tracked object (both represented as bounding boxes)
		This function returns a vector of shorts which will be one of the following values
		@param det The newly predicited detections
		@param trks The currently tracked objects
		@param -output matches, the indexes of matching det and trks
		@param -output unmatched_det The indexs of det that could not be matched	
	*/
	void associate_detections_to_trackers(const std::vector<cv::Rect>& det, 
										const std::vector<StateType>& trks,
										std::vector<std::pair<int,int> > & matches,
										std::vector<int>& unmatched_det);


	/** returns the IOU between to opencv Rect objects
	  @param rect1 
	  @param rect2
	  @return iou 
	*/
	double getIOU(const cv::Rect& r1, const cv::Rect& r2);
};

