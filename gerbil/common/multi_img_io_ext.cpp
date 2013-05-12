/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include <multi_img.h>
#include <opencv2/highgui/highgui.hpp>

#ifdef WITH_BOOST_FILESYSTEM
	#include "boost/filesystem.hpp"
#else
	#include <cerrno>
	#include <libgen.h>
	#include <sys/stat.h>
#endif

#include <fstream>

using namespace std;

void multi_img::read_image(const string& filename)
{
	pair<vector<string>, vector<BandDesc> > bands;

	// try to read in file list first
	bands = parse_filelist(filename);
	if (bands.first.empty()) {
		// maybe we got a .LAN image
		if (read_image_lan(filename))
			return;
		// maybe we got a single image as argument
		read_image(vector<string>(1, filename));
	} else {
		// read in multispectral image data
		read_image(bands.first, bands.second);
	}
}

struct hackstream : public std::ifstream {
	hackstream(const char* in, std::ios_base::openmode mode)	: ifstream(in, mode) {}
	void readus(unsigned short &d)
	{	read((char*)&d, sizeof(unsigned short));	}
	void readui(unsigned int &d)
	{	read((char*)&d, sizeof(unsigned int));		}
	void reada(unsigned char *a, size_t num)
	{	read((char*)a, sizeof(unsigned char)*num);	}
	void reada(unsigned short *a, size_t num)
	{	read((char*)a, sizeof(unsigned short)*num);	}
};

void multi_img::fill_bil(ifstream &in, unsigned short depth)
{
	/* read data in BIL (band interleaved by line) format */

	// depth == 0 -> unsigned char; depth == 2 -> unsigned short (from .lan)

	// we use either the 8bit or the 16bit array, have both for convenience. */
	unsigned char *srow8 = new unsigned char[width];
	unsigned short *srow16 = new unsigned short[width];
	for (int y = 0; y < height; ++y) {
		for (unsigned int d = 0; d < size(); ++d) {
			multi_img::Value *drow = bands[d][y];
			if (depth == 0)
				in.read((char*)srow8, sizeof(unsigned char)*width);
			else
				in.read((char*)srow16, sizeof(unsigned short)*width);
			for (int x = 0; x < width; ++x)
				drow[x] = (Value)(depth == 0 ? srow8[x] : srow16[x]);
		}
	}
	delete[] srow8;
	delete[] srow16;
	
	/* rescale data according to minval/maxval */
	Value srcmaxval = (depth == 0 ? (Value)255. : (Value)65535.);
	Value scale = (maxval - minval)/srcmaxval;
	for (unsigned int d = 0; d < size(); ++d) {
		if (minval != 0.) {
			bands[d] = bands[d] * scale + minval;
		} else {
			if (maxval != srcmaxval) // evil comp., doesn't hurt
				bands[d] *= scale;
		}
	}
}

bool multi_img::read_image_lan(const string& filename)
{
	// we omit checks for data consistency
	assert(empty());

	// parse header
	unsigned short depth, size;
	unsigned int rows, cols;
	hackstream in(filename.c_str(), ios::in | ios::binary);
	char buf[7] = "123456"; // enforce trailing \0
	in.read(buf, 6);
	if (strcmp(buf, "HEADER") && strcmp(buf, "HEAD74")) {
		return false;
	}

	in.readus(depth);
	in.readus(size);
	in.seekg(16);
	in.readui(cols);
	in.readui(rows);
	in.seekg(128);

	if (depth != 0 && depth != 2) {
		std::cout << "Data format not supported yet."
				" Please send us your file!" << std::endl;
		return false;
	}

	std::cout << "Total of " << size << " bands. "
	             "Spatial size: " << cols << "x" << rows
	          << "\t(" << (depth == 0 ? "8" : "16") << " bits)" << std::endl;

	// prepare image
	init(rows, cols, size);

	// read raw data
	fill_bil(in, depth);

	in.close();
	return true;
}

void multi_img::write_out(const string& base, bool normalize, bool in16bit) const
{
	// create directory
#ifdef WITH_BOOST_FILESYSTEM
	boost::filesystem::path basepath(base);
	bool success =
			boost::filesystem::is_directory(basepath) ||
			boost::filesystem::create_directory(basepath);
	if (!success) {
		std::cerr << "Writing failed! "
					 "Could not create directory " << base << std::endl;
		return;
	}
#include "boost/version.hpp"
#if BOOST_VERSION < 104600 // V2 API is default
	std::string filebase(basepath.filename()), dir(filebase);
#else
	std::string filebase(basepath.filename().string()), dir(filebase);
#endif
#elif __unix__
	int status = mkdir(base.c_str(), 0777);
	if (status != 0 && errno != EEXIST) {
		std::cerr << "Writing failed!"
					 "Could not create directory " << base << std::endl;
		return;
	}
	char *f = strdup(base.c_str()), *ff = basename(f);
	std::string filebase(ff), dir(filebase);
	free(f);
#else
	// don't use subdir
	std::string filebase(base.substr(base.find_last_of("/") + 1)), dir("./");
#endif

	// header of text file
	ofstream txtfile((base + ".txt").c_str());
	txtfile << size() << "\n";
	txtfile << dir << "\n";

	// preparation of scale and shift
	Value scale = (!normalize ? 1.f
	               : (in16bit ? (Value)65535.f/(maxval - minval)
				              : (Value)255.f/(maxval - minval)));
	Value shift = (!normalize ? 0.f : -scale*minval);

	// parameters to image writer
	vector<int> flags;
	flags.push_back(CV_IMWRITE_PNG_COMPRESSION);
	flags.push_back(9);  // [0-9] 9 being max compression, default is 3

	// write out band files and corresponding text file entries at once
	char name[1024];
	for (size_t i = 0; i < size(); ++i) {
		sprintf(name, "%s%02d.png", filebase.c_str(), (int)i);
		txtfile << name << " " << meta[i].rangeStart;
		if (meta[i].rangeStart != meta[i].rangeEnd) // print range, if available
			txtfile << " "  << meta[i].rangeEnd;
		txtfile << "\n";

		if (in16bit || normalize) { // data conversion needed
			cv::Mat output;
			bands[i].convertTo(output, (in16bit ? CV_16U : CV_8U),
							   scale, shift);
			cv::imwrite(base + "/" + name, output, flags);
		} else {
			cv::imwrite(base + "/" + name, bands[i], flags);
		}
	}

    txtfile.close();
}

// parse file list
pair<vector<string>, vector<multi_img::BandDesc> >
		multi_img::parse_filelist(const string& filename)
{
	pair<vector<string>, vector<BandDesc> > empty;

	ifstream in(filename.c_str());
	if (in.fail())
		return empty;

	unsigned int count;
	string base;
	in >> count;
	in >> base;
	if (in.fail())
		return empty;

#ifdef WITH_BOOST_FILESYSTEM
	boost::filesystem::path basepath(base), filepath(filename);
	if (!basepath.is_complete()) {
		basepath = filepath.remove_leaf() /= basepath;
		base = basepath.string();
	}
#elif __unix__
	if (base[0] != '/') {
		char *f = strdup(filename.c_str()), *d = dirname(f);
		base = string(d).append("/").append(base);
		free(f);
	}
#else
	std::cerr << "Warning: only absolute file paths accepted." << std::endl;
#endif
	base.append("/"); // TODO: check if o.k. in Windows
	stringstream in2;
	string fn; float a, b;
	vector<string> files;
	vector<BandDesc> descs;
	in >> ws;
	for (; count > 0; count--) {
		in2.clear();
		in.get(*in2.rdbuf()); in >> ws;
		in2 >> fn;
		if (in2.fail()) {
			cerr << "fail!" << endl;
			return empty;	 // file inconsistent -> screw it!
		}
		files.push_back(base + fn);

		in2 >> a;
		if (in2.fail()) { // no band desc given
			descs.push_back(BandDesc());
		} else {
			in2 >> b;
			if (in2.fail()) // only center filter wavelength given
				descs.push_back(BandDesc(a));
			else			// filter borders given
				descs.push_back(BandDesc(a, b));
		}
	}
	in.close();
	return make_pair(files, descs);
}
