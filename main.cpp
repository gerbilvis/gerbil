#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <highgui.h>
#include <qapplication.h>

#include "mfams.h"
#include "viewerwindow.h"

using namespace std;

void runFAMS(param_mfams& options, const multi_img& image, const string& base) {
	// load points
	FAMS cfams(options.use_LSH);
	cfams.ImportPoints(image);
	if (options.findKL) {
	// find K, L
		std::pair<int, int> ret = cfams.FindKL(options.Kmin, options.K,
										options.Kjump, options.L, options.k,
										options.bandwidth, options.epsilon);
		options.K = ret.first; options.L = ret.second;
		cerr << "Found K = " << options.k << "\tL = " << options.L << endl;
	} else {
	// actually run MS
		switch (options.starting) {
		case JUMP:
			cfams.RunFAMS(options.K, options.L, options.k, options.jump,
					  options.bandwidth, base);
			break;
		case PERCENT:
			cfams.RunFAMS(options.K, options.L, options.k, options.percent,
					  options.bandwidth, base);
			break;
		default:
			cfams.RunFAMS(options.K, options.L, options.k,
					  options.bandwidth, base);
		}

		if (!options.batch) {
			// save the data
			cfams.SaveModes(base);
			// save pruned modes
			cfams.SavePrunedModes(base);
			cfams.SaveMymodes(base);
/*			if (options.starting == ALL) {
				sprintf(tmp, "%s/%s.seg", options.outputdir.c_str(), options.inputfile.c_str());
				cfams.CreatePpm(tmp);//FIXME
			}*/
		}

		if (options.starting == ALL) {
			// save image which holds segment indices of each pixel
			IplImage *seg =	cfams.segmentImage(true);
			cvSaveImage((base + ".idx.png").c_str(), seg);
			cvReleaseImage(&seg);
		} else {
			cerr << "Note: As mean shift is not run on all input points, no "
					"output images were created." << endl;
		}
	}
}

int main(int argc, char** argv) {
	param_mfams options;
	
	if (!options.parse(argc, argv))
		exit(1);

	// load image	
	multi_img image(options.inputfile);
	if (image.empty())
		return 2;

/* Example for the multi_img API
	// get pixel at position 0, 0
	multi_img::Pixel a = image(0, 0);

	// convert pixel to cv::Mat
	cv::Mat_<multi_img::Value> b = image.Matrix(a);

	// alternatively: construct the matrix in one line
	b = image.Matrix(image(1, 1));

	// all values -> 255
	b.setTo(255.);

	// copy these values back in the multispectral image
	image.setPixel(1, 1, b);

	// quicker to type: do the same stuff as above by creating a pixel of the
	// required dimension, where all entries are initialized with 255
	image.setPixel(cv::Point(2, 2), multi_img::Pixel(image.size(), 255.));

	// work on a segment of the image:
	// this always requires a matrix containing a mask with the required
	// pixels. multi_img::Mask is internally a cv::Mat_<uchar>
	multi_img::Mask m(image.height, image.width, (uchar)0);

	// mark some pixels, in this example with a circle
	cv::circle(m, cv::Point(63, 63), 50, 255, 10);

	// now there are two possibilities to work with this segment: either call
	// getSegment, which returns a std::vector of pixel const pointers to the
	// selected image pixels. This is highly efficient, but const.
	// Second, it is possible to retrieve a copy of the pixels. These can be
	// arbitrarily manipulated.
	vector<multi_img::Pixel> v = image.getSegmentCopy(m);
	// thresholding example for the 0'th dimension (in this case, at 400nm)
	for (int i = 0; i < v.size(); ++i) {
		if (v[i][0] < 25.)
			v[i].assign(image.size(), image.minval);
		else
			v[i].assign(image.size(), image.maxval);
	}
	// copy the segment back in the image.
	image.setSegment(v, m);
*/
	//TODO: input filenames with path elements
	string base = options.outputdir + "/" + options.inputfile;

//	image.write_out(base + ".a");
	
	// log image data
	multi_img log = image.clone();
	log.apply_logarithm();
//	log.write_out(base + ".b");
	
	// compute spectral gradient
	multi_img gradient = log.spec_gradient();
//	gradient.write_out(base + ".c");

//	runFAMS(options, gradient, base);

	QApplication app(argc, argv);
	ViewerWindow window(image, gradient);
	window.show();
	return app.exec();
}

