#include "myCanny.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

void help()
{
	cout <<
			"\nDemonstrate Canny edge detection\n"
			"Call:\n"
			"./edge image_name\n"
			"./edge dx_name dy_name\n" << endl;
}

int edgeThresh1 = 1, edgeThresh2 = 3;
int sigma = 0;
double sigmaD = 0.0;
Mat image, dx, dy, gray, edge, cedge;
Mat dxBlur, dyBlur;
myCanny can;

// define a trackbar callback
void onTrackbar(int, void*)
{
	if (image.empty()) {
		
		sigmaD = (double)sigma/10.0;
// 		cv::GaussianBlur(dx, dxBlur, cvSize(3,3), sigmaD, sigmaD, BORDER_DEFAULT);
// 		cv::GaussianBlur(dy, dyBlur, cvSize(3,3), sigmaD, sigmaD, BORDER_DEFAULT);
// 		can.canny(dxBlur, dyBlur, edge, edgeThresh1, edgeThresh2, 3, true);
		dxBlur = dx.clone();
		dyBlur = dy.clone();
		if(sigma > 0)
		{
			cv::GaussianBlur(dxBlur, dxBlur, cvSize(3,3), sigmaD, sigmaD, BORDER_DEFAULT);
			cv::GaussianBlur(dyBlur, dyBlur, cvSize(3,3), sigmaD, sigmaD, BORDER_DEFAULT);
		}	
		can.canny(dxBlur, dyBlur, edge, edgeThresh1, edgeThresh2, 3, true);		
		imshow("DX", dxBlur);
		imshow("DY", dyBlur);
		
// 		can.canny(dx, dy, edge, edgeThresh1, edgeThresh2, 3, true);
	} else {
		blur(gray, edge, Size(3,3));

		// Run the edge detector on grayscale
		Canny(edge, edge, edgeThresh1, edgeThresh2, 3);
		cedge = Scalar::all(0);
		
		image.copyTo(cedge, edge);
		imshow("Edge map BGR", cedge);
	}
	imshow("Edge map", edge);
}

bool writeImage(string fn)
{
	size_t found = fn.find(".png");
	if(found != string::npos)
	{
		std::stringstream can;
		can << "_canny_" << edgeThresh1 << "_" << edgeThresh2 << "_s_"<< sigmaD;
		fn.insert(found, can.str());
	}
	else
	{	
		cerr <<"Invalid input file, expected PNG image" <<endl;
		return false;
	}	

	return imwrite(fn, edge);
}


int main(int argc, char** argv)
{
	if (argc == 2) {
	   	image = imread(argv[1], 1);
	   	if (!image.empty()) {
			cedge.create(image.size(), image.type());
			cvtColor(image, gray, CV_BGR2GRAY);
			namedWindow("Edge map BGR", 2);
	   	}
	}
	else if (argc == 3) {
		dx = imread(argv[1], -1);
   	dy = imread(argv[2], -1);
		namedWindow("DX", CV_GUI_EXPANDED);
		namedWindow("DY", CV_GUI_EXPANDED);
	}
	if (image.empty() && (dx.empty() || dy.empty()))
	{
		help();
		return -1;
	}

	namedWindow("Edge map", CV_GUI_EXPANDED);

	// create a toolbar
	createTrackbar("Sigma (x10)", "Edge map", &sigma, 50, onTrackbar);
	createTrackbar("Threshold 1", "Edge map", &edgeThresh1, 100, onTrackbar);
	createTrackbar("Threshold 2", "Edge map", &edgeThresh2, 100, onTrackbar);

	// Show the image
	onTrackbar(0, 0);

	// Wait for a key stroke; the same function arranges events processing
	waitKey(0);
	std::cout << "+++ Writing edge image +++" <<std::endl;
	if(writeImage(argv[1]))
		cout << "+++ Success +++"<< endl;
	else
		cout << "+++ Failed +++"<< endl;
	return 0;
}
