#ifndef MULTI_IMG_H
#define MULTI_IMG_H

#include <vector>
#include <cv.h>
#include <QImage>

// struct to hold a multispectral image
class multi_img {

public:
	typedef double Value;
	typedef cv::Mat_<Value> Band;
	typedef std::vector<Value> Pixel;

	/// default constructors
	multi_img() : width(0), height(0) {}
	multi_img(size_t size) : width(0), height(0), bands(size) {}

	/** reads in and processes either
		(a) one image file containing 1 or several color channels
		(b) a file containing file list (see read_filelist)
	*/
	multi_img(const std::string& filename);

	/// returns number of bands
	inline unsigned int size() const { return bands.size(); };

	/// returns true if image is uninitialized
	inline bool empty() const { return bands.empty(); };

	/// returns one band
	inline const Band& operator[](unsigned int band) const
	{ assert(band < size()); return bands[band]; }

	/// returns spectral data of a single pixel
	inline const Pixel& operator()(unsigned int row, unsigned int col) const
	{	assert(row < height && col < width);
		if (dirty(row, col))
			rebuildPixel(row, col);
		return pixels[row*width + col];
	};
	/// returns spectral data of a single pixel
	inline const Pixel& operator()(cv::Point pt) const
	{ return operator ()(pt.y, pt.x); }

	/// sets a single pixel
	void setPixel(unsigned int row, unsigned int col, const Pixel& values);
	/// sets a single pixel
	inline void setPixel(cv::Point pt, const Pixel& values)
	{ setPixel(pt.y, pt.x, values); }

	/// replaces a band
//	void setBand(unsigned int band, Band data);

	///	invalidate pixel cache (TODO: ROI) (maybe protected?)
	void resetPixels() const;

	/// rebuild whole pixel cache (don't wait for dirty pixel access)
	void rebuildPixels() const;

	/// rebuild a single pixel (inefficient if many pixels are processed)
	void rebuildPixel(unsigned int row, unsigned int col) const;

	/// returns pointer to data in interleaved format
	// you have to free it after use! KTHXBYE
	unsigned short* export_interleaved() const;

	/// return QImage of specific band
	QImage export_qt(unsigned int band) const;
	
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
	Value minval, maxval;
	/// ensuring image dimension consistency
	int width, height;

protected:
	inline void setDirty(unsigned int row, unsigned int col,
						 bool dirty = true) const;

	std::vector<Band> bands;
	mutable std::vector<Pixel> pixels;
	mutable cv::Mat_<uchar> dirty;
	mutable int dirtycount;
};

#endif
