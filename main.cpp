#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <highgui.h>
#include "mfams.h"

using namespace std;

/* mfams main function */

int main(int argc, char** argv) {
	param_mfams options;
	
	if (!options.parse(argc, argv))
		exit(1);

	// load image	
	multi_img image(options.inputfile);
	if (image.empty())
		return 2;

	image.write_out(options.outputdir + "/a");
	
	// log image data
	image.apply_logarithm();
	image.write_out(options.outputdir + "/b");
	
	// compute spectral gradient
	multi_img gradient = image.spec_gradient();
	gradient.write_out(options.outputdir + "/c");
	

	// load points
	FAMS cfams(options.use_LSH);
	cfams.ImportPoints(gradient);

	if (options.findKL) {
	// find K, L
		std::pair<int, int> ret = cfams.FindKL(options.Kmin, options.K,
										options.Kjump, options.L, options.k,
										options.bandwidth, options.epsilon);
		options.K = ret.first; options.L = ret.second;
		bgLog("Found K = %d L = %d (write them down)\n", options.K, options.L);
	} else {
	// actually run MS	
		switch (options.starting) {
		case JUMP:
			cfams.RunFAMS(options.K, options.L, options.k, options.jump,
					  options.bandwidth, options.outputdir, options.inputfile);
			break;
		case PERCENT:
			cfams.RunFAMS(options.K, options.L, options.k, options.percent,
					  options.bandwidth, options.outputdir, options.inputfile);
			break;
		default:
			cfams.RunFAMS(options.K, options.L, options.k,
					  options.bandwidth, options.outputdir, options.inputfile);
		}

		if (options.starting != ALL)
			cerr << "Note: As mean shift is not run on all input points, no "
			        "output images are created." << endl;
		if (!options.batch) {
			// save the data
			cfams.SaveModes(options.outputdir, options.inputfile);
			// save pruned modes
			cfams.SavePrunedModes(options.outputdir, options.inputfile);
			cfams.SaveMymodes(options.outputdir, options.inputfile);
			if (options.starting == ALL) {
				sprintf(tmp, "%s/%s.seg", options.outputdir.c_str(), options.inputfile.c_str());
				cfams.CreatePpm(tmp);//FIXME
			}
		}
		// save image which holds segment indices of each pixel
		sprintf(tmp, "%s/%s", options.outputdir.c_str(), options.inputfile.c_str());
		cfams.SaveSegments(tmp);//FIXME
	}
}



/* main function to test multi_img */
int main2(int argc, char **argv) {
	
	if (argc < 2)
		return 1;

	multi_img image(argv[1]);
	
	if (image.empty())
		return 2;

	image.write_out("/tmp/blackjack/a");
	
	// log image data
	image.apply_logarithm();
	image.write_out("/tmp/blackjack/b");
	
	// compute spectral gradient
	multi_img gradient = image.spec_gradient();
	gradient.write_out("/tmp/blackjack/c");
	
	if (argc > 2)
		gradient.export_interleaved();
	
	// we could clean up here ;-)
	
	return 0;
}
