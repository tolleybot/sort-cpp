#include "Sort.h"
#include <cmath>

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
void Sort::associate_detections_to_trackers(const std::vector<cv::Rect>& det,
	const std::vector<StateType>& trks,
	std::vector<std::pair<int, int> > & matches,
	std::vector<int>& unmatched_det,
	std::vector<int>& unmatched_trks)
{

}
// ----------------------------------------------------------------------------------
const std::vector<TrackingBox>& Sort::update(std::vector<cv::Rect>& det)
{
	frame_count++;
	// used to mark items to remove
	vector<int> to_del;	
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
	vector<int> unmatched_trks;
	// attempted to associated new det and tracked
	associate_detections_to_trackers(det, trks, matching, unmatched_det, unmatched_trks);

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

	for (size_t i = 0, j = trackers.size() - 1; i < trackers.size(); i++, j--)
	{
		KalmanTracker* pTrk = trackers[j];

		if (pTrk->m_time_since_update < 1 && (pTrk->m_hit_streak >= min_hits || frame_count <= min_hits)) {

			TrackingBox tb = { frame_count, pTrk->m_id,  pTrk->get_state() };
			results.push_back(tb);
		}				
	}

	return results;
}
