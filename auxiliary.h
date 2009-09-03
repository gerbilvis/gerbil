#ifndef AUXILIARY_H
#define AUXILIARY_H

#ifndef __max
#define __max(a, b)    (a > b ? a : b)
#define __min(a, b)    (a < b ? a : b)
#endif

#include <boost/program_options.hpp>

/* parameters */

struct param {
	param();

	// image to be processed
	std::string inputfile;
	// working directory
	std::string outputdir;
	
	// batch mode: only output end result, no other files for further analysis
	bool batch;
	
	boost::program_options::options_description all, global;
};

enum ms_points {
	ALL,
	JUMP,
	PERCENT
};
// to use this enum in configuration, following has to be defined
// see also ENUM_MAGIC in .cpp file
#define ms_pointsString {"ALL", "JUMP", "PERCENT"}

struct param_mfams : public param {
	param_mfams();
	// read in values from command line
	bool parse(int argc, char** argv);

	bool use_LSH;
	int K, L; // LSH parameters
	
	// pilot density
	int k; // number of neighbors used for construction
	
	// starting points
	ms_points starting;
	int jump;
	float percent; 
	float bandwidth;
	
	// find optimal K and L automatically
	bool findKL;
	int Kmin, Kjump;
	float epsilon;
	
	boost::program_options::options_description mfams;
};


// color conversions in rgbluv.cpp
void rgb2hsv(float r, float g, float b, float *H, float *S, float *V);
void rgb2luv(float *RGB, float *LUV, int size);
void luv2rgb(float *RGB, float *LUV, int size);

// aux functions
void bgLog(const char *varStr, ...);

void timer_start();
double timer_stop();
double timer_elapsed(int prnt);


/* this is some macro trickery (just leave it as is) to make ENUMs
   work for reading (program_options) and writing (to stream) */
#define ENUM_MAGIC(ENUM) \
	const char* ENUM ## Str[] = ENUM ## String;\
	void validate(boost::any& v, const std::vector<std::string>& values, \
	               ENUM* target_type, int) \
	{ \
		boost::program_options::validators::check_first_occurrence(v); \
		const std::string& s = \
			boost::program_options::validators::get_single_string(values); \
		for (unsigned int i = 0; i < sizeof(ENUM ## Str)/4; ++i) { \
			if (strcmp(ENUM ## Str[i], s.c_str()) == 0) { \
				v = boost::any((ENUM)i); \
				return; \
			} \
		} \
		throw boost::program_options::validation_error("invalid value"); \
	} \
	std::ostream& operator<<(std::ostream& o, ENUM e)  \
	{	o << ENUM ## Str[e]; return o;  }

#endif
