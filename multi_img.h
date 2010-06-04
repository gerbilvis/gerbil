#ifndef MULTI_IMG_H
#define MULTI_IMG_H

#include <vector>
#include <cv.h>
#ifdef VOLE_GUI
	class QImage;
#endif

// struct to hold a multispectral image
class multi_img {

public:
	/// value type (may be changed to float for less precision)
	typedef double Value;
	typedef cv::Mat_<Value> Band;
	typedef cv::MatIterator_<Value> BandIt;
	typedef cv::MatConstIterator_<Value> BandConstIt;
	typedef std::vector<Value> Pixel;
	typedef cv::Mat_<uchar> Mask;
	typedef cv::MatIterator_<uchar> MaskIt;
	typedef cv::MatConstIterator_<uchar> MaskConstIt;

	/// struct that holds optional filter information for each band
	struct BandDesc {
		BandDesc() : center(0.f), rangeStart(0.f), rangeEnd(0.f), empty(true){}
		BandDesc(float c) : center(c), rangeStart(c), rangeEnd(c), empty(false){}
		BandDesc(float s, float e) : center((e + s)*0.5f),
									 rangeStart(s), rangeEnd(e), empty(false) {}
		/// center wavelength of the filter in nm
		float center;
		/// range of the filter (approximate filter edges in nm)
		float rangeStart, rangeEnd;
		/// filter information available (empty == false) or not
		bool empty;
	};

	/// default constructors
	multi_img() : width(0), height(0) {}
	multi_img(size_t size) : width(0), height(0), bands(size) {}

	/** reads in and processes either
		(a) an image file containing one or several color channels
		(b) a descriptor file that contains a file list (see read_filelist)
	*/
	multi_img(const std::string& filename);

	/// returns number of bands
	inline size_t size() const { return bands.size(); };

	/// returns true if image is uninitialized
	inline bool empty() const { return bands.empty(); };

	/// returns one band
	inline const Band& operator[](unsigned int band) const
	{ assert(band < size()); return bands[band]; }

	/// returns spectral data of a single pixel
	inline const Pixel& operator()(unsigned int row, unsigned int col) const
	{	assert((int)row < height && (int)col < width);
		if (dirty(row, col))
			rebuildPixel(row, col);
		return pixels[row*width + col];
	};
	/// returns spectral data of a single pixel
	inline const Pixel& operator()(cv::Point pt) const
	{ return operator ()(pt.y, pt.x); }

	/// returns spectral data of a segment (using mask)
	std::vector<const Pixel*> getSegment(const Mask &mask);
	/// returns copied spectral data of a segment (using mask)
	std::vector<Pixel> getSegmentCopy(const Mask &mask);

	/// sets a single pixel
	void setPixel(unsigned int row, unsigned int col, const Pixel& values);
	/// sets a single pixel
	void setPixel(unsigned int row, unsigned int col,
				  const cv::Mat_<Value>& values);
	/// sets a single pixel
	inline void setPixel(cv::Point pt, const Pixel& values)
	{ setPixel(pt.y, pt.x, values); }
	/// sets a single pixel
	inline void setPixel(cv::Point pt, const cv::Mat_<Value>& values)
	{ setPixel(pt.y, pt.x, values); }

	/// replaces a band with optional mask
	void setBand(unsigned int band, const Band &data,
				 const Mask &mask = Mask());

	/// replaces all pixels in mask with given values
	/**
	  @arg values vector of pixel values which must hold the same amount of
		   members as non-null mask values, ordered by row index first, column
		   index second
	 */
	void setSegment(const std::vector<Pixel> &values, const Mask& mask);
	void setSegment(const std::vector<cv::Mat_<Value> > &values,
					const Mask& mask);

	///	invalidate pixel cache (TODO: ROI) (maybe protected?)
	void resetPixels() const;

	/// rebuild whole pixel cache (don't wait for dirty pixel access)
	void rebuildPixels() const;

	/// rebuild a single pixel (inefficient if many pixels are processed)
	void rebuildPixel(unsigned int row, unsigned int col) const;

	/// returns pointer to data in interleaved format
	// you have to free it after use! KTHXBYE
	unsigned short* export_interleaved() const;

#ifdef VOLE_GUI
	/// return QImage of specific band
	QImage export_qt(unsigned int band) const;
#endif
	
	/// get independent copy of image
	/**
		@arg cloneCache if true, copies pixel cache. if false, cache is not
			 copied and all pixels are dirty (useful if cache unneeded or will
			 be invalidated anyway)
	 */
	multi_img clone(bool cloneCache = true);
	
	/// compile image from filelist (files can have several channels
	// will not erase previous data
	void read_image(const std::vector<std::string> &files,
					const std::vector<BandDesc> &descs = std::vector<BandDesc>());

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
	static std::pair<std::vector<std::string>, std::vector<BandDesc> >
			read_filelist(const std::string& filename);
	
	/// copies Pixel into a OpenCV matrix (row vector)
	/* The copy is needed as there is no "ConstMatrix" type.
	   Note that this is just a wrapper to OpenCV functionality, but it
	   ensures that you are doing it "right". */
	inline static cv::Mat_<Value> Matrix(const Pixel& p)
	{ return cv::Mat_<Value>(p, true); }

	/// copies Matrix into a Pixel
	inline static Pixel Matrix(const cv::Mat_<Value>& m)
	{ // see setPixel function
	  // return Pixel(m.begin(), m.end());
		Pixel ret(m.rows*m.cols); BandConstIt it; int i;
		for (i = 0, it = m.begin(); it != m.end(); ++it, ++i)
			ret[i] = *it;
		return ret;
	}

/* finally some variables */
	/// minimum and maximum values (by data format, not actually observed data!)
	Value minval, maxval;
	/// ensuring image dimension consistency
	// signed int because cv::Mat.{rows, cols} are also signed int
	int width, height;
	/// band meta-data
	std::vector<BandDesc> meta;
protected:
	std::vector<Band> bands;
	mutable std::vector<Pixel> pixels;
	mutable Mask dirty;
};

#endif
