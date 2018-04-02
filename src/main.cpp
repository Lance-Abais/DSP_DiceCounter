#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

class Dice
{
    public:
        Dice(Rect bound, vector<KeyPoint> pip){
            this->boundary = bound;
            this->pips = pip;
        }
        Rect Getboundary() { return boundary; }
        void Setboundary(Rect val) { boundary = val; }
        vector<KeyPoint> Getpips() { return pips; }
        void Setpips(vector<KeyPoint> val) { pips = val; }
        int countPip(){return pips.size();}

    private:
        Rect boundary;
        vector<KeyPoint> pips;
};


Mat ori,blurred,mag,grad_dir;
int thresh = 150;
int max_thresh = 255;
RNG rng(12345);



string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void GradientMagnitude(Mat input)
{
      Mat grad_x,grad_y;
      Sobel(blurred, grad_x,CV_64F,1,0,3,1,0,BORDER_CONSTANT);
      Sobel(blurred, grad_y,CV_64F,0,1,3,1,0,BORDER_CONSTANT);
      phase(grad_x,grad_y,grad_dir,true);
      for(int i=1; i<grad_dir.rows-1; i++){
        for(int j=1; j<grad_dir.cols-1; j++){
            char thisAngle = grad_dir.at<float>(i,j);
            thisAngle = thisAngle;
            if(thisAngle<45) grad_dir.at<float>(i,j) = 0;
            else if(thisAngle>=45||thisAngle<90) grad_dir.at<float>(i,j) = 45;
            else if(thisAngle<135||thisAngle>=90) grad_dir.at<float>(i,j) = 90;
            else grad_dir.at<float>(i,j) = 135;

        }
      }
      Laplacian(input,mag,CV_32FC3,3,3,0,BORDER_CONSTANT);


}

Mat NonMax_Suppression(Mat input){

      // NAIVE PIXEL ITERATION
      int count =0;
      Mat output = input;
      for(int i=1; i<grad_dir.rows-1; i++){
        for(int j=1; j<grad_dir.cols-1; j++){
            //Directions
            const int east = j + 1;
            const int north = i - 1;
            const int south = i + 1;
            const int west = j - 1;

            char dir = grad_dir.at<char>(i,j);
            char magH = input.at<char>(i,j);

            //Neighborhood
            char magE = input.at<char>(i,east);
            char magW = input.at<char>(i,west);
            char magN = input.at<char>(north,j);
            char magS = input.at<char>(south,j);
            char magNE = input.at<char>(north,east);
            char magSW = input.at<char>(south,west);
            char magSE = input.at<char>(south,east);
            char magNW = input.at<char>(north,west);
            if(magH!=0)
            {
                if(dir == 0 && ((magE>magH) || (magW>magH) || magNE>magH || magSW>magH || magNW>magH || magSE>magH) || magH<magE+magNE || magH<magW+magSW) magH = 0;
                if(dir == 45 && (magNE>magH || magSW>magH || magN>magH || magS>magH || magE>magH || magW>magH || magH<magNE+magN || magH<magSW+magS)) magH = 0;
                if(dir == 90 && (magN>magH || magS>magH || magNW>magH || magNE>magH || magSE>magH || magSW>magH || magH<magW+magNW || magH<magE+magSE)) magH = 0;
                if(dir == 135 && (magNW>magH || magSE>magH || magN>magH || magS>magH || magE>magH || magW>magH || magH<magN+magNW || magH<magSE+magS)) magH = 0;
            }
                output.at<char>(i,j) = magH;
        }
      };
    return output;
}

Mat HysteresisThresholding(Mat input, char maxG, char minG){

    Mat out = input;
    const char MAX_BRIGHTNESS = 255;


    for(int i=1; i<input.rows-1;i++){
    for(int j=1; j<input.cols-1;j++){
    char magH = input.at<char>(i,j);
    //Directions
    const int east = j + 1;
    const int north = i - 1;
    const int south = i + 1;
    const int west = j - 1;

    if(magH>=maxG)
    {
    out.at<char>(i,j) = MAX_BRIGHTNESS;
    char magE = input.at<char>(i,east);
    char magW = input.at<char>(i,west);
    char magN = input.at<char>(north,j);
    char magS = input.at<char>(south,j);
    char magNE = input.at<char>(north,east);
    char magSW = input.at<char>(south,west);
    char magSE = input.at<char>(south,east);
    char magNW = input.at<char>(north,west);

    if(magE>=minG) magE = MAX_BRIGHTNESS;
    if(magW>=minG) magW = MAX_BRIGHTNESS;
    if(magN>=minG) magN = MAX_BRIGHTNESS;
    if(magS>=minG) magS = MAX_BRIGHTNESS;
    if(magNE>=minG) magNE = MAX_BRIGHTNESS;
    if(magSE>=minG) magSE = MAX_BRIGHTNESS;
    if(magNW>=minG) magNW = MAX_BRIGHTNESS;
    if(magSW>=minG) magSW = MAX_BRIGHTNESS;

    out.at<char>(i,east) = magE;
    out.at<char>(i,west) = magW;
    out.at<char>(south,j) = magS;
    out.at<char>(north,j) = magN;
    out.at<char>(south,east) = magSE;
    out.at<char>(south,west) = magSW;
    out.at<char>(north,west) = magNW;
    }
}
}

return out;

}

Mat water(Mat input){

    Mat src = input;

    for( int x = 0; x < src.rows; x++ ) {
      for( int y = 0; y < src.cols; y++ ) {
          if ( src.at<Vec3b>(x, y) == Vec3b(255,255,255) ) {
            src.at<Vec3b>(x, y)[0] = 0;
            src.at<Vec3b>(x, y)[1] = 0;
            src.at<Vec3b>(x, y)[2] = 0;
          }
        }
    }

    Mat kernel = (Mat_<float>(3,3) <<
            1,  1, 1,
            1, -8, 1,
            1,  1, 1);
    Mat imgLaplacian;
    Mat sharp = src;
    filter2D(sharp, imgLaplacian, CV_32F, kernel);
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    src = imgResult;

    Mat bw;
    cvtColor(src, bw, CV_BGR2GRAY);
    threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    Mat dist;
    distanceTransform(bw, dist, CV_DIST_L2, 3);

    normalize(dist, dist, 0, 1., NORM_MINMAX);

    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);

    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);

    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);

    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);


    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    //imshow("Markers", markers*10000);

    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);

    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }

    imshow("Final Result", dst);

    return dst;
}


Mat cannyEdge(Mat input, char maxG, char minG){

imshow("Before", input);
cvtColor(input,input,CV_BGR2GRAY);
input.convertTo(input,CV_8UC3);
GaussianBlur( input, blurred, Size( 5, 5 ), 0, 0 );
blurred.convertTo(blurred,CV_8UC3);
GradientMagnitude(blurred);
Mat suppressed = NonMax_Suppression(mag);
Mat out = HysteresisThresholding(suppressed, maxG,minG);
out.convertTo(out,CV_8UC3);
imshow("Canny",out);
return out;
}

vector<KeyPoint> pipCount(Mat input, Rect boundRect){
    SimpleBlobDetector::Params params;


    params.filterByInertia = true;
    params.minInertiaRatio = 0.6;


    vector<KeyPoint> keypoints;
    vector<KeyPoint> tempKey;

    Ptr<SimpleBlobDetector> blobDetector = SimpleBlobDetector::create(params);
    blobDetector->detect(input, keypoints);
    for(int i = 0; i<keypoints.size(); i++)
    {
        if(boundRect.contains(keypoints[i].pt)) tempKey.push_back(keypoints[i]);
    }
    keypoints = tempKey;

    return keypoints;
}


int main(int argc, char** argv)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> tempCon;
    vector<Dice> dice;
    vector<Rect> diceBounds;
    string filename = "";
    cout<<"Enter filename: ";
    cin>>filename;
    namedWindow("Before" , CV_WINDOW_AUTOSIZE);
    ori = imread(filename, IMREAD_COLOR);
    if(ori.empty()){
        cout<<"There is no file with that name. \n";
        return 1;
    }
    Mat output = cannyEdge(ori, 120, 20);

    //WATERSHED
    Mat watershed;
    watershed = water(ori);
    cvtColor(watershed,watershed,CV_BGR2GRAY);

    findContours( watershed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Mat drawcont = Mat::zeros(watershed.size(), CV_8UC3);
    for( int i = 0; i< contours.size(); i++ ){
           Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
           drawContours( drawcont, contours, i, color, 2, 8, hierarchy, 0, Point() );
           diceBounds.push_back(boundingRect(contours[i]));
         }


    for(int i = 0; i<diceBounds.size(); i++)
    {
        rectangle(ori,diceBounds[i],Scalar(0,0,255),1,LINE_8);
        dice.push_back(Dice(diceBounds[i],pipCount(output,diceBounds[i])));
    }
    imshow("bounded dice",ori);

    vector<Dice> finalDiceCount;
    for(int i = 0; i<dice.size(); i++){
        if(dice[i].countPip()>0) finalDiceCount.push_back(dice[i]);
    }

    cout<<"Number of Dice: "<<finalDiceCount.size()<<endl;

    for(int i = 0; i<finalDiceCount.size(); i++)
        cout<<"Dice number: "<<i+1<<"|"<<"Pip Count: "<<finalDiceCount[i].countPip()<<endl;
    waitKey(0);
    return 0;

}
