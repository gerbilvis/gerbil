#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <highgui.h>
#include "multi_img.h"

using namespace std;


multi_img::multi_img(const string& filename) : width(0), height(0) {
	pair<vector<string>, vector<BandDesc> > bands;
	
	// try to read in file list first
	bands = read_filelist(filename);
	if (bands.first.empty()) {
		// maybe we got a single image as argument
		read_image(vector<string>(1, filename));
	} else {
		// read in multispectral image data
		read_image(bands.first, bands.second);
	}
};

multi_img multi_img::clone(bool cloneCache) {
	multi_img ret(size());
	ret.minval = minval; ret.maxval = maxval;
	ret.width = width; ret.height = height;
	for (int i = 0; i < size(); ++i)
		ret.bands[i] = bands[i].clone();

	if (cloneCache) {
		ret.pixels = pixels;
		ret.dirty = dirty.clone();
	} else	ret.resetPixels();

	return ret;
}

void multi_img::resetPixels() const {
	if (pixels.empty())
		pixels.resize(width * height, Pixel(size()));
	if (dirty.empty())
		dirty = Mask(width, height, 255);
	else
		dirty.setTo(255);
}

void multi_img::rebuildPixels() const {
	BandConstIt it;
	register int d, i;
	for (d = 0; d < size(); ++d) {
		const Band &src = bands[d];
		for (it = src.begin(), i = 0; it != src.end(); ++it, ++i)
			pixels[i][d] = *it;
	}
	dirty.setTo(0);
}

void multi_img::rebuildPixel(unsigned int row, unsigned int col) const {
	Pixel &p = pixels[row*width + col];
	for (int i = 0; i < size(); ++i)
		p[i] = bands[i](row, col);

	dirty(row, col) = 0;
}

std::vector<const multi_img::Pixel*> multi_img::getSegment(const Mask &mask) {
	assert(mask.rows == height && mask.cols == width);

	std::vector<const Pixel*> ret;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				if (dirty(row, col))
					rebuildPixel(row, col);
				ret.push_back(&pixels[row*width + col]);
			}
		}
	}
	return ret;
}

std::vector<multi_img::Pixel> multi_img::getSegmentCopy(const Mask &mask) {
	assert(mask.rows == height && mask.cols == width);

	std::vector<Pixel> ret;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				if (dirty(row, col))
					rebuildPixel(row, col);
				ret.push_back(pixels[row*width + col]);
			}
		}
	}
	return ret;
}

void multi_img::setPixel(unsigned int row, unsigned int col,
						 const Pixel &values) {
	assert(row < height && col < width);
	assert(values.size() == size());
	Pixel &p = pixels[row*width + col];
	p = values;
	for (int i = 0; i < size(); ++i)
		bands[i](row, col) = p[i];

	dirty(row, col) = 0;
}

// matrix version
void multi_img::setPixel(unsigned int row, unsigned int col,
						 const cv::Mat_<Value>& values) {
	assert(row < height && col < width);
	assert(values.rows*values.cols == size());
	Pixel &p = pixels[row*width + col];
	/* should work, but bug in OpenCV (ConstIterator is STL-incompatible)
	   https://code.ros.org/trac/opencv/ticket/321
	p.assign(values.begin(), values.end()); */
	// workaround
	BandConstIt it; int i;
	for (i = 0, it = values.begin(); it != values.end(); ++it, ++i)
		p[i] = *it;
	// end workaround

	for (int i = 0; i < size(); ++i)
		bands[i](row, col) = p[i];

	dirty(row, col) = 0;
}

void multi_img::setBand(unsigned int band, const Band &data,
						const Mask &mask) {
	assert(band < size());
	assert(data.rows == height && data.cols == width);
	Band &b = bands[band];
	BandConstIt bit = b.begin();
	MaskConstIt dit = dirty.begin();
	/* we use opencv to copy the band data. afterwards, we update the pixels
	   cache. we do this only for pixels, which are not dirty yet (and would
	   need a complete rebuild anyways. As we instantly fix the other pixels,
	   those do not get marked as dirty by us. */
	if (mask.empty()) {
		assert(mask.rows == height && mask.cols == width);
		MaskConstIt mit = mask.begin();
		data.copyTo(b, mask);
		for (int i = 0; bit != b.end(); ++bit, ++dit, ++mit, ++i)
			if ((*mit > 0)&&(*dit == 0))
				pixels[i][band] = *bit;
	} else {
		data.copyTo(b);
		for (int i = 0; bit != b.end(); ++bit, ++dit, ++i) {
			if ((*dit == 0))
				pixels[i][band] = *bit;
		}
	}
}

void multi_img::setSegment(const std::vector<Pixel> &values, const Mask &mask) {
	assert(mask.rows == height && mask.cols == width);
	int i = 0;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				setPixel(row, col, values[i]);
				++i;
			}
		}
	}
}

void multi_img::setSegment(const std::vector<cv::Mat_<Value> > &values,
						   const Mask &mask) {
	assert(mask.rows == height && mask.cols == width);
	int i = 0;
	for (int row = 0; row < height; ++row) {
		const uchar *m = mask[row];
		for (int col = 0; col < width; ++col) {
			if (m[col] > 0) {
				setPixel(row, col, values[i]);
				++i;
			}
		}
	}
}

// TODO: rewrite using Pixel vector
unsigned short* multi_img::export_interleaved() const {
	Value scale = 65535.0/(maxval - minval);
	unsigned short *ret = new unsigned short[size()*width*height];
	
	/* actually we don't care about the ordering of the pixels, just all
	   values have to get in there in interleaved format */
	BandConstIt it;
	register int d, i;
	for (d = 0; d < size(); ++d) {
		const Band &src = (*this)[d];
		// i starts with offset d, then jumps over dimensions
		for (it = src.begin(), i = d; it != src.end(); ++it, i += size())
			ret[i] = (unsigned short)((*it - minval) * scale);
	}
	
	return ret;
}

// exports one band
QImage multi_img::export_qt(unsigned int band) const
{
	assert(band < size());
	Value scale = 255.0/(maxval - minval);
	const Band &src = bands[band];
	QImage dest(width, height, QImage::Format_ARGB32);
	unsigned int color = 0;
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			color = (unsigned int)((src[y][x] - minval) * scale);
			dest.setPixel(x, y, qRgba(color, color, color, 255));
		}
	}
	return dest;
}

// parse file list
pair<vector<string>, vector<multi_img::BandDesc> >
		multi_img::read_filelist(const string& filename) {
	pair<vector<string>, vector<BandDesc> > empty;
	ifstream in(filename.c_str());
	if (in.fail())
		return empty;

	int count;
	string base;
	in >> count;
	in >> base;
	if (in.fail())
		return empty;

	base.append("/");
	stringstream in2;
	string fn; float a, b;
	vector<string> files;
	vector<BandDesc> descs;
	cout << count << "\t" << base << endl;
	for (; count > 0; count--) {
/*		in.get(*in2.rdbuf());
		cout << in2.str() << endl;
		in.get(*in2.rdbuf());
		cout << in2.str() << endl;*/
		in >> fn;
		if (in.fail())
			return empty;	 // file inconsistent -> screw it!
		files.push_back(base + fn);

/*		in2 >> a;
		if (in2.fail()) { // no band desc given
			descs.push_back(BandDesc());
		} else {
			in2 >> b;
			if (in2.fail()) // only center filter wavelength given
				descs.push_back(BandDesc(a));
			else			// filter borders given
				descs.push_back(BandDesc(a, b));
		}*/
	}
	in.close();
	return make_pair(files, descs);
}

// read multires. image into vector
void multi_img::read_image(const vector<string> &files, const vector<BandDesc> &descs) {
	/* our favorite range */
	minval = 0.; maxval = 255.;

	for (int fi = 0; fi < files.size(); ++fi) {
	    cv::Mat src = cv::imread(files[fi].c_str(), -1); // flag -1: preserve format
	    
		if (src.empty()) {
			cerr << "ERROR: Failed to load " << files[fi] << endl;
			continue;
		}
		
		// write or test size
		if (width > 0) {
			if (src.cols != width || src.rows != height) {
				cerr << "ERROR: Size mismatch for image " << files[fi] << endl;
				continue;
			}
		} else {
			width = src.cols;
			height = src.rows;
		}
		
		// we only expect CV_8U or CV_16U right now
		Value srcmaxval = (src.depth() == CV_16U ? 65535. : 255.);

		// operate on sane data type from now on
		cv::Mat_<Value> img = src;
		// rescale data accordingly
		if (maxval != srcmaxval) // evil comp., doesn't hurt
			img *= maxval/srcmaxval;
		
		// split image
		std::vector<cv::Mat> channels;
		cv::split(img, channels);
				
		// add everything in at the end
		bands.insert(bands.end(), channels.begin(), channels.end());
		
	    cout << "Added " << files[fi] << ":\t" << channels.size()
             << (channels.size() == 1 ? " channel, " : " channels, ")
             << (src.depth() == CV_16U ? 16 : 8) << " bits" << endl;
	}

	/* invalidate pixel cache as pixel length has changed
	   This step is _mandatory_ also to initialize cache containers */
	pixels.clear();
	resetPixels();

	/* add meta information if present. */
	if (!descs.empty()) {
		assert(meta.size() + descs.size() == size());
		meta.insert(meta.end(), descs.begin(), descs.end());
	} else {
		meta.resize(size());
	}

	cout << "Total of " << size() << " dimensions.";
	cout << "\tSpacial size: " << width << "x" << height << endl;
}

void multi_img::write_out(const string& base, bool normalize) {
	char name[1024];
	for (int i = 0; i < size(); ++i) {
		sprintf(name, "%s%02d.png", base.c_str(), i);

		if (normalize) {
			cv::Mat_<uchar> normalized;
			Value scale = 255./(maxval - minval);
			bands[i].convertTo(normalized, CV_8U, scale, -scale*minval);
			cv::imwrite(name, normalized);
		} else
			cv::imwrite(name, bands[i]);
	}
}

void multi_img::apply_logarithm() {
	for (int i = 0; i < size(); ++i) {
		// will assign large negative value to 0 pixels
		cv::log(bands[i], bands[i]);
		// get rid of negative values (when pixel value was 0)
		cv::max(bands[i], 0., bands[i]);
	}
	// data format has changed as follows
	minval = 0.;
	maxval = log(maxval);
	// cache became invalid
	resetPixels();
}

multi_img multi_img::spec_gradient() {
	multi_img ret(size() - 1);
	// data format of output
	ret.minval = -maxval/2.;
	ret.maxval =  maxval/2.;
	ret.width = width; ret.height = height;

	for (int i = 0; i < size()-1; ++i) {
		ret.bands[i] = (bands[i+1] - bands[i]);
	}
	ret.resetPixels();
	return ret;
}
