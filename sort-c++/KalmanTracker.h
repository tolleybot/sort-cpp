///////////////////////////////////////////////////////////////////////////////
// KalmanTracker.h: KalmanTracker Class Declaration

#ifndef KALMAN_H
#define KALMAN_H 2

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>

#define StateType cv::Rect_<float>

/** This class represents the internel state of 
individual tracked objects observed as bounding box.
*/
class KalmanTracker
{

public:
	KalmanTracker()
	{
		init_kf(StateType());		
		m_id = kf_count;
	}
	KalmanTracker(StateType initRect)
	{
		init_kf(initRect);		
		m_id = kf_count;
		kf_count++;
	}

	~KalmanTracker()
	{
		m_history.clear();
	}

	/** make next prediction
	*/
	StateType predict();
	/** update kalaman filter with new
		prediction 
	*/
	void update(StateType stateMat);
	/** get current state
	*/
	StateType get_state();	

	static int kf_count;
	int m_time_since_update;
	int m_hits;
	int m_hit_streak;
	int m_age;
	int m_id;

private:
	/// converts from center bb to tl bb anchor
	StateType get_rect_xysr(float cx, float cy, float s, float r);
	/// performs initialization
	void init_kf(StateType stateMat);

private:
	/// OpenCV kalaman filter object	
	cv::KalmanFilter kf;
	/// Used to updated the current measurement
	cv::Mat measurement;
	/// the last prediction
	StateType m_lastPred;
};




#endif