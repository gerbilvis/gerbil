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

