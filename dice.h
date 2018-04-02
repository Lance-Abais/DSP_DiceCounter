#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>

#ifndef DICE_H
#define DICE_H

using namespace cv;
using namespace std;

class Dice
{
	public:
		Dice(Mat);
		~Dice();
		int count_pips(Mat, vector<KeyPoint>&);
		void get_location(Rect);
		
    private:
        vector<KeyPoint> keypoints;
        vector<Point> location;
        Rect boundary;
        Mat output;
};


#endif // DICE_H
