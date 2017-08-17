///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <set>
#include <vector>
#include "Sort.h"

using namespace std;
using namespace cv;

// global variables for counting
#define CNUM 20
double total_time = 0.0;


/** formats data to make it usable by sort
 *
 * @param filename Full filepath to data file
 * @param max_frame The last frame in the sequence
 * @param detFrameDat - output - all tracking boxes per frame
 * @return true if successful
 */
bool readDataFile(string filename, int & max_frame, vector<vector<cv::Rect> >& detFrameDat)
{
    // 1. read detection file
    ifstream detectionFile(filename);


    if (!detectionFile.is_open())
    {
        cerr << "Error: can not find file " << filename << endl;
        return false;
    }

    string detLine;

    char ch;
    float tpx, tpy, tpw, tph;
    istringstream ss;
    vector<TrackingBox> vtb;

    while ( getline(detectionFile, detLine) )
    {
        TrackingBox tb;
        ss.str(detLine);
        ss >> tb.frame >> ch >> tb.id >> ch;
        ss >> tpx >> ch >> tpy >> ch >> tpw >> ch >> tph;
        ss.str("");

        tb.box = Rect_<float>(Point_<float>(tpx, tpy), Point_<float>(tpx + tpw, tpy + tph));
        vtb.push_back(tb);
    }

    detectionFile.close();


    // 2. group data by frame
    int maxFrame = 0;
    for (auto tb : vtb) // find max frame number
    {
        if (maxFrame < tb.frame)
            maxFrame = tb.frame;
    }

    vector<vector<TrackingBox>> dets;

    for (int fi = 0; fi < maxFrame; fi++)
    {
        vector<TrackingBox> tempVec;

        for (auto tb : vtb) {
            if (tb.frame == fi + 1) // frame num starts from 1
                tempVec.push_back(tb);
        }

        dets.push_back(tempVec);
    }
}

// ----------------------------------------------------------------------------------------------

void TestSORT(string seqName)
{
	cout << "Processing " << seqName << "..." << endl;

    string detFileName = "data/" + seqName + "/det.txt";

    vector<vector<cv::Rect>> dets;

    int max_frame = 0;

    if (!readDataFile(detFileName, max_frame, dets))
    {
        return;
    }

	// prepare result file.
	ofstream resultsFile;
	string resFileName = "output/" + seqName + ".txt";
	resultsFile.open(resFileName);

	if (!resultsFile.is_open())
	{
		cerr << "Error: can not create file " << resFileName << endl;
		return;
	}

    Sort sort_tracker;


	//////////////////////////////////////////////
	// main loop
	for (int fi = 0; fi < max_frame; fi++)
	{
		int64 start_time = getTickCount();

        // update for this frame
        const vector<TrackingBox> results = sort_tracker.update(dets[fi]);

		double cycle_time = (double)(getTickCount() - start_time);
		total_time += cycle_time / getTickFrequency();

		for (auto tb : results)
			resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;


	}

	resultsFile.close();

}


/** main function
*/
int main()
{
	vector<string> sequences = { "PETS09-S2L1", "TUD-Campus", "TUD-Stadtmitte", "ETH-Bahnhof", "ETH-Sunnyday", "ETH-Pedcross2", "KITTI-13", "KITTI-17", "ADL-Rundle-6", "ADL-Rundle-8", "Venice-2" };
	for (auto seq : sequences)
		TestSORT(seq);

	// Note: time counted here is of tracking procedure, while the running speed bottleneck is opening and parsing detectionFile.
 // cout << "Total Tracking took: " << total_time << " for " << total_frames << " frames or " << ((double)total_frames / (double)total_time) << " FPS" << endl;

	return 0;
}

