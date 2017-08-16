#include "Sort.h"
#include <cmath>
#include <limits>
#include "Hungarian.h"

using namespace std;

// ----------------------------------------------------------------------------------
Sort::Sort()
{
	reset();
}
// ----------------------------------------------------------------------------------
Sort::~Sort()
{
	reset();
}
// ----------------------------------------------------------------------------------
void Sort::reset()
{
	frame_count = -1;

	for (size_t i; i < trackers.size(); i++)
	{
		delete trackers[i];
	}
	trackers.clear();
}
// ----------------------------------------------------------------------------------
double Sort::getIOU(const cv::Rect& r1, const cv::Rect& r2)
{
	float in = (r1 & r2).area();
	float un = r1.area() + r2.area() - in;

	if (un < std::numeric_limits<double>::epsilon())
		return 0;

	return (double)(in / un);
}
// ----------------------------------------------------------------------------------
void Sort::associate_detections_to_trackers(const std::vector<cv::Rect>& det,
	const std::vector<StateType>& trks,
	std::vector<std::pair<int, int> > & matches,
	std::vector<int>& unmatched_det)
{
	// if there are not trackers then there are no matches
	if (trks.empty())
	{
		unmatched_det = det;	
		return;	
	}

	size_t detNum = det.size();
	size_t trkNum = trks.size();

	// create matching matrix, <det, trk> iou's
	vector<vector<double>> iouMatrix(trkNum, vector<double>(detNum, 0));
	
	for (size_t i=0; i < detNum; i++)
	{
		double* ptr = iouMatrix.ptr<double>(i);

		for (size_t j=0; j < trkNum; j++)
		{
		   ptr[j] = getIOU(det[i], trks[j]);
		}
	}

	// solve the assignment problem using hungarian algorithm.
	HungarianAlgorithm HungAlgo;
	vector<int> matched_indices;
	HungAlgo.Solve(iouMatrix, matched_indices);

	/// find all unmatched detections
	unmatched_det.clear();
	{
		// create an index of det
		for (size_t i = 0; i < detNum; i++)
			allItems.insert(i);

		for (size_t i = 0; i < trkNum; ++i)
			matchedItems.insert(matched_indices[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatched_det, unmatched_det.begin()));
	}

	// TODO: DON stopped here

}
// ----------------------------------------------------------------------------------
const std::vector<TrackingBox>& Sort::update(std::vector<cv::Rect>& det)
{
	frame_count++;

	// clear last results
	results.clear();

	vector<StateType> trks;

	// loop through existing tracks
	for (size_t i = 0; i < trackers.size(); i++)
	{
		// make latest prediction
		StateType state = trackers[i]->predict;
		// make sure we have a valid state
		if (std::isnan(state.x) || std::isnan(state.y) || std::isnan(state.width) || std::isnan(state.height))
			continue

			trks.push_back(state);
	}

	vector<pair<int, int> > matching;
	vector<int> unmatched_det;	
	// attempted to associated new det and tracked
	associate_detections_to_trackers(det, trks, matching, unmatched_det);

	// update matched trackers with assigned detections
	for (size_t i = 0; i < matching.size(); i++)
	{
		trackers->update(trks[matching[i].first]);
	}

	// create and initialise new trackers for unmatched detections
	for (size_t i = 0; i < unmatched_det.size(); i++)
	{
		trackers.push_back(new KalmanTracker(trks[unmatched_det[i]]));
	}

	vector<int> to_del;
	for (size_t i = trackers.size() - 1; i >=0; i--)
	{
		KalmanTracker* pTrk = trackers[i];

		if (pTrk->m_time_since_update < 1 && (pTrk->m_hit_streak >= min_hits || frame_count <= min_hits)) {

			TrackingBox tb = { frame_count, pTrk->m_id,  pTrk->get_state() };
			results.push_back(tb);
		}	

		// remove dead tracks
		if (pTrk->m_time_since_update > max_age)	
		{
			to_del.push_back(i)			
		}		
	}

	for (size_t i = 0 ; i < to_del.size(); i++)
	{
		trackers.erase(trackers.begin() + to_del[i])
	}

	return results;
}
