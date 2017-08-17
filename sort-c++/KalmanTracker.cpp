///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.cpp: KalmanTracker Class Implementation Declaration

#include "KalmanTracker.h"

using namespace std;
using namespace cv;

int KalmanTracker::kf_count = 0;


// initialize Kalman filter
void KalmanTracker::init_kf(StateType stateMat)
{
	int stateNum = 7;
	int measureNum = 4;
	kf = KalmanFilter(stateNum, measureNum, 0);

	measurement = Mat::zeros(measureNum, 1, CV_32F);

	kf.transitionMatrix = (Mat_<float>(stateNum, stateNum) <<
		1, 0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 1, 0,
		0, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1);

	kf.measurementMatrix = (Mat_<float>(measureNum, stateNum) <<
		1, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0);

    // Q
	setIdentity(kf.processNoiseCov);

	kf.processNoiseCov(Range(4,stateNum-1),
    	               Range(4,stateNum-1)) *= .01f;
	

    // R
	setIdentity(kf.measurementNoiseCov);
	kf.measurementNoiseCov(Range(2,measureNum-1),
    	                   Range(2,measureNum-1)) *= 10.0f;

    // P
	setIdentity(kf.errorCovPost, Scalar::all(10));
	kf.errorCovPost(Range(4,measureNum-1),
    	                   Range(4,measureNum-1)) *= 1000.0f;
	
	// initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;

	m_time_since_update = 0;
	m_hits = 0;
	m_hit_streak = 0;
	m_age = 0;
}


// Predict the estimated bounding box.
StateType KalmanTracker::predict()
{
	// predict
	Mat p = kf.predict();
	m_age += 1;

	if (m_time_since_update > 0)
		m_hit_streak = 0;
	m_time_since_update += 1;

	return get_rect_xysr(p.at<float>(0, 0), p.at<float>(1, 0), p.at<float>(2, 0), p.at<float>(3, 0));
}

// Update the state vector with observed bounding box.
void KalmanTracker::update(StateType stateMat)
{
	m_time_since_update = 0;
	m_hits += 1;
	m_hit_streak += 1;

	// measurement
	measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	measurement.at<float>(2, 0) = stateMat.area();
	measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

	// update
	kf.correct(measurement);
}


// Return the current state vector
StateType KalmanTracker::get_state()
{
	Mat s = kf.statePost;
	return get_rect_xysr(s.at<float>(0, 0), s.at<float>(1, 0), s.at<float>(2, 0), s.at<float>(3, 0));
}


// Convert bounding box from [cx,cy,s,r] to [x,y,w,h] style.
StateType KalmanTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
	float w = sqrt(s * r);
	float h = s / w;
	float x = (cx - w / 2);
	float y = (cy - h / 2);

	if (x < 0 && cx > 0)
		x = 0;
	if (y < 0 && cy > 0)
		y = 0;

	return StateType(x, y, w, h);
}


