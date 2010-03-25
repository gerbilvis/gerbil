#ifndef MULTI_IMG_H
#define MULTI_IMG_H

#include <vector>
#include <cv.h>
#include <QImage>

// struct to hold a multispectral image
struct multi_img : public std::vector<cv::Mat_<double> > {
public:
	/// default constructors
	multi_img() : width(0), height(0) {}
	multi_img(size_t size) : width(0), height(0), std::vector<cv::Mat_<double> >(size) {}

	/** reads in and processes either
		(a) one image file containing 1 or several color channels
		(b) a file containing file list (see read_filelist)
	*/
	multi_img(const std::string& filename);

	/// returns pointer to data in interleaved format
	// you have to free it after use! KTHXBYE
	unsigned short* export_interleaved() const;

	/// return QImage of specific dimension
	QImage export_qt(int dim) const;
	
	/// get independent copy of image
	// use case: apply log on image but keep original version as well
	multi_img clone();
	
	/// compile image from filelist (files can have several channels
	// will not erase previous data (use cleanup for that)
	void read_image(std::vector<std::string>& files);

	/// write the whole image with base name base (may include directories)
	/* output is 8 bit grayscale PNG image
	   if normalize (default), output is scaled/shifted for better conversion */
	void write_out(const std::string& base, bool normalize = true);

/* here the real fun starts */
	/// apply natural logarithm on image
	void apply_logarithm();
	
	/// return spectral gradient of image (note: apply log. first!)
	multi_img spec_gradient();


/* helper functions */
	/// reads a file list for multispectral image
	/* file format:
		number_of_files(int)	common_path(string)
		filename(string)
		...
		filename(string)
	*/
	static std::vector<std::string> read_filelist(const std::string& filename);
	
	/// minimum and maximum values (by data format, not actually observed data!)
	double minval, maxval;
	/// ensuring image dimension consistency
	int width, height;
};

#endif
