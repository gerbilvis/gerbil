/*
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MULTI_IMG_H
#define MULTI_IMG_H

/** default range of image data. for custom range, set minval and maxval
	members before reading images */
#define MULTI_IMG_MIN_DEFAULT 0.
#define MULTI_IMG_MAX_DEFAULT 255.

#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include <cfloat>
#include <vector>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef WITH_BOOST
	#include <boost/shared_ptr.hpp>
#endif
#ifdef WITH_QT
	class QImage;
#endif

class Illuminant;

// FIXME what a mess
class Grad;
class Log;
class Clamp;
class Illumination;
class PcaProjection;
class MultiImg2BandMat;
class GradientCuda;
class GradientTbb;
class ClampTbb;
class ClampCuda;
class IlluminantTbb;
class IlluminantCuda;
class DataRangeTbb;
class DataRangeCuda;
class PcaTbb;

#define MULTI_IMG_FRIENDS \
	friend class RebuildPixels;\
	friend class ApplyCache;\
	friend class DetermineRange;\
	friend class Band2QImageTbb;\
	friend class RescaleTbb;\
	friend class Resize; \
	friend class Grad;\
	friend class Log;\
	friend class Clamp;\
	friend class Illumination;\
	friend class PcaProjection;\
	friend class MultiImg2BandMat;\
	friend class GradientCuda;\
	friend class GradientTbb;\
	friend class ClampTbb;\
	friend class ClampCuda;\
	friend class IlluminantTbb;\
	friend class IlluminantCuda;\
	friend class DataRangeTbb;\
	friend class DataRangeCuda;\
	friend class PcaTbb;

class multi_img_base {
public:

/** @name Storage types
 *  These are types for convenience that are based on a single type
	choice for multispectral data (Value). You should only use the Value
	type for computation that is directly processing the image data. It is
	not a global decision on precision, only on image data precision.
 */
//@{

	/// value type (use float to save memory, double for higher precision)
	typedef float Value;
	struct Range {
		Range() : min(), max() {}
		Range(Value min, Value max) : min(min), max(max) {}
		Value min, max;
	};
	static const float ValueMin;
	static const float ValueMax;
	static const int ValueType = CV_32F;
	/// a spectral band
	typedef cv::Mat_<Value> Band;

	/// struct that holds optional filter information for each band
	struct BandDesc {
		BandDesc() : center(0.f), rangeStart(0.f), rangeEnd(0.f), empty(true){}
		BandDesc(float c) : center(c), rangeStart(c), rangeEnd(c), empty(false){}
		BandDesc(float s, float e) : center((e + s)*0.5f),
		                             rangeStart(s), rangeEnd(e), empty(false) {}
		inline std::string str() const {
			if (empty)
				return std::string();
			std::ostringstream str;
			if (rangeStart == rangeEnd)
				str << center << " nm";
			else
				str << rangeStart << " nm - " << rangeEnd << " nm";
			return str.str();
		}
		/// center wavelength of the filter in nm
		float center;
		/// range of the filter (approximate filter edges in nm)
		float rangeStart, rangeEnd;
		/// filter information available (empty == false) or not
		bool empty;
	};

//@}

	/// default constructor
	multi_img_base()
		: minval(MULTI_IMG_MIN_DEFAULT), maxval(MULTI_IMG_MAX_DEFAULT),
		  width(0), height(0) {}

	/// barebone constructor
	multi_img_base(unsigned int size)
		: minval(MULTI_IMG_MIN_DEFAULT), maxval(MULTI_IMG_MAX_DEFAULT),
		  width(0), height(0), meta(size) {}

	/// copy constructor
	multi_img_base(const multi_img_base &a) : minval(a.minval), maxval(a.maxval), 
		width(a.width), height(a.height), meta(a.meta) {}

	virtual ~multi_img_base() {}

	/// returns number of bands
	virtual unsigned int size() const = 0;

	/// returns true if image is uninitialized
	virtual bool empty() const = 0;	

	/// returns one band
    virtual void getBand(size_t band, Band &data) const = 0;

	/// returns the roi part of the given band
	virtual void scopeBand(const Band &source, const cv::Rect &roi, Band &target) const = 0;

	/// minimum and maximum values (by data format, not actually observed data!)
	Value minval, maxval;

	/** spatial dimensionality
		ensuring consistency over all bands
		@note signed int because cv::Mat.{rows, cols} are also signed int
	 **/
	int width;
	/** spatial dimensionality
		ensuring consistency over all bands
		@note signed int because cv::Mat.{rows, cols} are also signed int
	 **/
	int height;

	/// band meta-data
	std::vector<BandDesc> meta;


protected:

	MULTI_IMG_FRIENDS
};

/// Class that holds a multispectral image.
/**
	This class holds image data ranging from a single grayscale image to a
	hyperspectral image. Each frequency band (or image channel) is held in a single
	OpenCV Matrix. A caching mechanism is employed that also allows access to data
	on a per-pixel level (interleaved storage).

	@note There is extended functionality implemented in multi_img_{io,ext,io_ext}.cpp.
	These are functions of sole interest for true multispectral images,
	while all functionality implemented inside Vole may also be useful for ordinary RGB images.

  */
class multi_img : public multi_img_base {

public:

/** @name Storage types
 *  These are types for convenience that are based on a single type
	choice for multispectral data (Value). You should only use the Value
	type for computation that is directly processing the image data. It is
	not a global decision on precision, only on image data precision.
 */
//@{

#ifdef WITH_BOOST
	/// for convenience: a multi_img shared ptr
	typedef boost::shared_ptr<multi_img> ptr;
#endif

	/// a spectral pixel.
	/** @note Pixel will always be a std::vector. You can count on this. **/
	typedef std::vector<Value> Pixel;

//@}

	enum NormMode {
		NORM_OBSERVED = 0,
		NORM_THEORETICAL = 1,
		NORM_FIXED = 2
	};


/** @name Constructors & Copy/Assignment **/
//@{

	/// default constructor
	multi_img() : multi_img_base() {}

	/// barebone constructor
	multi_img(unsigned int size)
		: multi_img_base(size), roi(0, 0, 0, 0), bands(size) {}

	/// empty image constructor (to create synthetic images)
	multi_img(int height, int width, unsigned int size);

	/// copy constructor
	multi_img(const multi_img &);

	/// copy a spatial region of interest
	multi_img(const multi_img_base &a, const cv::Rect &roi);

	/// copy a subrange of the spectrum (including both ends)
	multi_img(const multi_img &a, unsigned int start, unsigned int end);

	/// assignment operator
	/** @note A copy of the image is created.
              Use clone() if you want to preserve the cache. **/
	multi_img & operator=(const multi_img &);

	/** reads in and processes either
		(a) an image file containing one or several color channels
		(b) a descriptor file that contains a file list (see read_filelist)
	*/
	multi_img(const std::string& filename);
	/** compiles image from cv::Mat (every channel becomes a band)
		source range is automatically determined by format */
	multi_img(const cv::Mat& image);
	/** compiles image from cv::Mat (every channel becomes a band) */
	multi_img(const cv::Mat& image, Value srcmin, Value srcmax);

	/// get independent copy of image
	/**
		@arg cloneCache if true, copies pixel cache. if false, cache is not
			 copied and all pixels are dirty (useful if cache unneeded or will
			 be invalidated anyway, common use case for doing a copy)
	 */
	multi_img clone(bool cloneCache = false) const;

	/// virtual destructor, does nothing
	virtual ~multi_img() {}

//@}

/** @name Element access operators for reading **/
//@{

	/// returns number of bands
	virtual unsigned int size() const;

	/// returns true if image is uninitialized
	virtual bool empty() const;

	/// returns one band
    virtual void getBand(size_t band, Band &data) const;

	/// returns the roi part of the given band
	virtual void scopeBand(const Band &source, const cv::Rect &roi, Band &target) const;

	/// returns one band
	inline const Band& operator[](unsigned int band) const
	{ assert(band < size()); return bands[band]; }

	/// returns spectral data of a single pixel
	inline const Pixel& operator()(unsigned int row, unsigned int col) const
	{	assert((int)row < height && (int)col < width);
		if (anydirt && dirty(row, col))
			rebuildPixel(row, col);
		return pixels[row*width + col];
	}

	/// returns spectral data of a single pixel
	inline const Pixel& operator()(cv::Point pt) const
	{ return operator ()(pt.y, pt.x); }

	/// returns spectral data of a single pixel (only if *no* pixel is dirty!)
	inline const Pixel& atIndex(unsigned int idx) const
	{	assert(!anydirt);
		return pixels[idx];
	}

	/// returns spectral data of a segment (using mask)
	std::vector<const Pixel*> getSegment(const cv::Mat1b &mask);
	/// returns copied spectral data of a segment (using mask)
	std::vector<Pixel> getSegmentCopy(const cv::Mat1b &mask);

//@}

/** @name Element access operators for writing **/
//@{

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
				 const cv::Mat1b &mask = cv::Mat1b());

	/// replaces all pixels in mask with given values
	/**
	  @arg values vector of pixel values which must hold the same amount of
		   members as non-null mask values, ordered by row index first, column
		   index second
	 */
	void setSegment(const std::vector<Pixel> &values, const cv::Mat1b& mask);
	void setSegment(const std::vector<cv::Mat_<Value> > &values,
					const cv::Mat1b& mask);

	/// initialize image data with a spectral vector
	void setTo(const Pixel& p);

	//@}

	/** @name Pixel cache operations **/
	//@{

	///	invalidate pixel cache
	/** @arg force rebuild even if structure was present before
		@todo ROI, maybe change to protected
	*/
	void resetPixels(bool force = false) const;

	/// rebuild whole pixel cache (don't wait for dirty pixel access)
	/** if optimistic, it is checked first if the cache is already sane.
		set optimistic to false if you know beforehand it is dirty. */
	void rebuildPixels(bool optimistic = true) const;

	/// rebuild a single pixel (inefficient if many pixels are processed)
	void rebuildPixel(unsigned int row, unsigned int col) const;

//@}

/** @name Data export and conversion **/
//@{
	/// composes OpenCV Matrix from the image (no data copy!)
	cv::Mat Mat() {
		cv::Mat ret;
		cv::merge(&bands[0], size(), ret);
		return ret;
	}

	/// returns pointer to data in interleaved format
	/** @note You have to free the new data array after use.
		@param useDataRange If this is true, normalization will be done
				based on the actual data range instead of minval and maxval
	**/
	unsigned short* export_interleaved(bool useDataRange = false) const;

#ifdef WITH_QT
	/// return QImage of specific band
	QImage export_qt(unsigned int band) const;
#endif

	/// return sRGB color space representation of the image
	cv::Mat_<cv::Vec3f> bgr() const;

	/// return sRGB color space representation of a multispectral pixel
	cv::Vec3f bgr(const Pixel &p) const;
	static cv::Vec3f bgr(const Pixel &p,
		size_t dim, const std::vector<BandDesc> &meta, Value maxval);

//@}

/** @name Pixel <-> Matrix **/
//@{

	/// copies Pixel into a OpenCV matrix (row vector)
	/* The copy is needed as there is no "ConstMat_" type.
	   Note that this is just a wrapper to OpenCV functionality, but it
	   ensures that you are doing it "right". */
	inline static cv::Mat_<Value> toMat(const Pixel& p)
	{ return cv::Mat_<Value>(p, true); }

	/// copies Matrix into a Pixel
	inline static Pixel toPixel(const cv::Mat_<Value>& m)
	{ return Pixel(m.begin(), m.end()); }
//@}

/** @name Reading and writing from/to other formats/files **/
//@{

	/// helper for read_image, add data from one cv::Mat, returns #channels
	int read_mat(const cv::Mat &src);
	/// add data from one cv::Mat with given source range, returns #channels
	int read_mat(const cv::Mat &src, Value srcmin, Value srcmax);

	/// compile image from filelist (files can have several channels)
	// will not erase previous data
	void read_image(const std::vector<std::string> &files,
					const std::vector<BandDesc> &descs = std::vector<BandDesc>());

	/// fill image with raw data from file stream, used by read_image_lan
	/** @note Part of Gerbil. **/
	void fill_bil(std::ifstream &in, unsigned short depth);

	/// helper for read_image for LAN images, returns true on success
	/** @note Part of Gerbil. **/
	bool read_image_lan(const std::string& filename);

	/// read grayscale, RGB, LAN or filelist image
	/** @note Without gerbil, only grayscale and RGB is supported. **/
	void read_image(const std::string& filename);

	/// write the whole image with base name base (may include directories)
	/** Output is 8 bit or 16 bit grayscale PNG image.
	    @param normalize If set (default), output is scaled/shifted for better conversion.
		@param in16bit If set (default), use 16 bits for storage, otherwise 8 bits
		@note This function is only available in Gerbil. If you use this
		      class in Vole for RGB data, use Mat() and then imwrite().
	**/
	void write_out(const std::string& base, bool normalize = true, bool in16bit = true) const;

//@}

/** @name Data statistics **/
//@{
	/// determine minimum, maximum of observed data
	/** Helps to find discrepancy between data and theoretical minval/maxval.
	   @param fraction If this is > 0, histogramming is employed to
	           find a range such as atmost fraction of the data values lie
	           outside the range. This is useful to ignore outliers that
	           would inordinately stretch the range.
	**/
	Range data_range(double fraction = 0.) const;

	/// compute PCA of the image
	/**
	  @param components number of components to compute (if 0, compute #bands)
	  **/
	cv::PCA pca(unsigned int components = 0) const;

	/// apply PCA transform to the image
	multi_img project(const cv::PCA &pca) const;

	/// apply PCA transform to a single vector (convenience function)
	static Pixel project(const Pixel& p, const cv::PCA& pca);

	/// apply inverse PCA transform to a single vector (convenience function)
	static Pixel backProject(const Pixel& p, const cv::PCA& pca);

//@}

/** @name Geometric transformations **/
//@{
	/// apply cv::flip() on the image
	/**
		@param flipcode 0: x-axis, 1: y-axis, -1: both (refer to OpenCV manual)
	  */
	void flip(int flipCode);

	/// apply cv::transpose() on the image
	void transpose();
//@}

/** @name Data manipulation **/
//@{
	/// clamp values that lie outside (minval, maxval) to minval/maxval
	/** Use this function if your algorithm depends on a guaranteed value range. **/
	void clamp();

	/// rescale image according to a given range
	/** The current minval & maxval are used to determine starting range.
        minval/maxval will then be updated to new range.
	  @note This function is to convert already existing data from one value
            range to another. If you know your preferred range before reading,
            you can set minval and maxval before calling read_image().
	  */
	void data_rescale(Value minval, Value maxval);

	/// stretch data such that it uses whole minval/maxval range.
	/** @note uses data_range() and data_rescale() functions. **/
	void data_stretch();

	/// stretch each band separately
	/**
   @param minval optional data range (default: image's minval/maxval are used)
   @param maxval optional data range (default: image's minval/maxval are used)
	  **/
	void data_stretch_single(Value minval = 0., Value maxval = 0.);

	/// apply natural logarithm on image
	void apply_logarithm();

	/// blur the image with gaussian kernel for noise removal
	void blur(cv::Size ksize, double sigmaX, double sigmaY = 0,
			  int borderType = cv::BORDER_DEFAULT);

	/// return spectral gradient of log. image
	/** @note: Apply the logarithm first! **/
	multi_img spec_gradient() const;

	/// return a copy with fewer bands (linear interpolation)
    multi_img spec_rescale(unsigned int newsize) const;
//@}

/** @name Helper functions **/
//@{

	/// reads a file list for multispectral image
	/** file format:
		number_of_files(int)	common_path(string)
		filename(string)	freq.start freq.end
		filename(string)	freq.center
		...
		filename(string)
	*/
	static std::pair<std::vector<std::string>, std::vector<BandDesc> >
			parse_filelist(const std::string& filename);

	/// helper function to create XYZ color space representation
	/// of a multispectral pixel
	void pixel2xyz(const Pixel &p, cv::Vec3f &xyz) const;
	static void pixel2xyz(const Pixel &p, cv::Vec3f &xyz, 
		size_t dim, const std::vector<BandDesc> &meta, Value maxval);

	/// helper function to do conversion from xyz to sRGB color space
	static void xyz2bgr(const cv::Vec3f &xyz, cv::Vec3f &rgb);
//@}

/** @name Illumination **/
//@{
	/// apply illuminant to the image (or remove)
	void apply_illuminant(const Illuminant&, bool remove = false);

	/// returns all illuminant coefficients relevant for this image
	std::vector<Value> getIllumCoeff(const Illuminant&) const;
//@}

	/// ROI associated with image data
	cv::Rect roi;
	
protected:
	/// write back pixel cache into band data
	void applyCache();

	/// simple data structure initialization
	void init(int height, int width, unsigned int size,
			  Value minval = MULTI_IMG_MIN_DEFAULT,
			  Value maxval = MULTI_IMG_MAX_DEFAULT);

	std::vector<Band> bands;
	mutable std::vector<Pixel> pixels;
	mutable cv::Mat1b dirty;
	mutable bool anydirt;

	MULTI_IMG_FRIENDS
};

#endif // opencv
#endif // multi_img.h
