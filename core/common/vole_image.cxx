/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLEIMAGE_CXX
#define VOLEIMAGE_CXX

#include <sstream>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <cstring>
#include "pixel.h"
#include "vole_image.h"

template <typename T>
VoleImage<T>::VoleImage(std::string the_filename) {
	filename = the_filename;
	array = 0;
	readImageWithoutCleaning(filename);
}

template <typename T>
VoleImage<T>::VoleImage(const VoleImage<T> &copy) {
	array = 0;
	allocate(copy.getMaxY(), copy.getMaxX());
	blobSize = copy.getBlobSize();
	std::memcpy(array, copy.getBlob(), blobSize * sizeof(T));

	filename = "";
}

template <typename T>
VoleImage<T>::VoleImage(int maxY, int maxX) {
	array = 0;
	allocate(maxY, maxX);
}

template <typename T>
VoleImage<T>::~VoleImage() {
	if (array != 0) {
		delete[] array;
		array = 0;
	}
//	free(array);
}

// FIXME: The read/write workflow is fine for TIFF images, but apparently not
// for PPM images - what can be the reason for this?
template <typename T>
void VoleImage<T>::writeImage(std::string filename) {

	Magick::Geometry geom(maxX, maxY);
	Magick::Image image(geom, "Black");
	std::cout << "writing " << filename << " with size " << image.rows() << " und " << image.columns() << std::endl;
	voleImageToMagick(*this, image);
	image.write(filename);
}

template <typename T>
void VoleImage<T>::writePPM(std::string filename, bool raw = true) {
	// find maximum color value
	unsigned int max = 0;
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			T &pixel = *(array + y*maxX + x);
			for (int i = 0; i < 3; i++) {
				if (pixel[i] > max) max = pixel[i];
			}
		}
	}
	assert(max < 65536);
	std::ofstream out;
	if (raw) {
		out.open(filename.c_str(), std::ios::binary);
		out << "P6 " << maxX << " " << maxY << " ";
		if (max < 256) { // encode everything in 1 byte
			out << "255" << std::endl; // color depth
			for (int y = 0; y < maxY; y++) {
				for (int x = 0; x < maxX; x++) {
					T &pixel = *(array + y*maxX + x);
					for (int i = 0; i < 3; i++) {
						out << static_cast<unsigned char>(pixel[i]);
					}
				}
			//	out << std::endl;
			}
		} else { // encode everything in 2 byte
			out << "65535" << std::endl; // color depth
			unsigned char tmp1, tmp2;
			unsigned int val;
			for (int y = 0; y < maxY; y++) {
				for (int x = 0; x < maxX; x++) {
					T &pixel = *(array + y*maxX + x);
					for (int i = 0; i < 3; i++) {
						val = pixel[i];
						tmp1 = val >> 8;
						tmp2 = val & 255;
						out << tmp1 << tmp2;
					}
				}
			//	out << std::endl;
			}
		}
	} else { // ASCII output
		out.open(filename.c_str());
		out << "P3 " << maxX << " " << maxY << " " << max << std::endl;
		for (int y = 0; y < maxY; y++) {
			for (int x = 0; x < maxX; x++) {
				T &pixel = *(array + y*maxX + x);
					out << static_cast<unsigned short>(pixel[0]) << " " << static_cast<unsigned short>(pixel[1]) << " " << static_cast<unsigned short>(pixel[2]);
				out << std::endl;
			}
		}
	}

	out.close();
}

template <typename T>
T &VoleImage<T>::operator()(int y, int x) {
	assert(array != 0);
	return *(array + y*maxX + x);
}


template <typename T>
void VoleImage<T>::getRegion(VoleImage<T> &region, int minY, int minX, int maxY, int maxX) {
	assert(minY >= 0);
	assert(minX >= 0);
	assert(maxY <= this->maxY);
	assert(maxX <= this->maxX);
	assert(maxY > minY);
	assert(maxX > minX);

	region.allocate(maxY-minY, maxX-minX);
	T *blob = region.getBlob();
	int ry = 0;
	int dimx = maxX - minX;
	for (int y = minY; y < maxY; y++) {
		memcpy(blob + ry*dimx, array + y*this->maxX + minX, dimx*sizeof(T));
		ry++;
	}
}


template <typename T>
void VoleImage<T>::magickToVoleImage(Magick::Image &mag, VoleImage<T> &img) {

	// TODO: Some asserts here to ensure that both images have equal size
	int maxX = img.getMaxX();
	int maxY = img.getMaxY();

	T *blob = img.getBlob();
	unsigned int channelDepth = blob[0].getChannelDepth();
	Magick::PixelPacket *magick_idx;
	int idx;

	double scaling = 1.0;
//	FIXME the following line does not work: mag->depth() gives not the correct values!
//	int shift = 8*channelDepth - mag->depth();
//	FIXME setting default import depth to 16 Bit - how can this be automated?
	int shift = 8*channelDepth - 16;
	if (shift < 0) {
		shift = -shift;
		scaling /= static_cast<double>(1 << shift);
	} else {
		scaling = 1 << shift;
	}

	// copy image in internal memory
	Magick::PixelPacket *pixels = mag.getPixels(0, 0, maxX, maxY);
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			idx = y*maxX + x;
			magick_idx = pixels + idx;

			blob[idx].set(0, magick_idx->red*scaling);
			blob[idx].set(1, magick_idx->green*scaling);
			blob[idx].set(2, magick_idx->blue*scaling);
			blob[idx].setActive();
		}
	}
}


template <typename T>
void VoleImage<T>::voleImageToMagick(VoleImage<T> &img, Magick::Image &mag) {
	int maxX = img.getMaxX();
	int maxY = img.getMaxY();

	// TODO: Some asserts here to ensure that both images have equal size

	mag.modifyImage();
	Magick::PixelPacket *pixels = mag.getPixels(0, 0, maxX, maxY);
	T *blob = img.getBlob();
	unsigned int channelDepth = blob[0].getChannelDepth();

	std::cout << "channeldepth = " << channelDepth << std::endl;

	double scaling = 1.0;
//	FIXME the following line does not work: mag->depth() gives not the correct values!
//	int shift = mag->depth() - 8*channelDepth;
//	FIXME setting default import depth to 16 Bit - how can this be automated?
	int shift = 16 - 8*channelDepth;
	if (shift < 0) {
		shift = -shift;
		scaling /= static_cast<double>(1 << shift);
	} else
		scaling = 1 << shift;

	// copy array contents into image
	int idx;
	Magick::PixelPacket *magick_idx;
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			idx = y*maxX + x;
			magick_idx = pixels + idx;
			T &vole_pixel = *(blob + idx);
			magick_idx->red   = static_cast<unsigned short int>(vole_pixel[0] * scaling);
			magick_idx->green = static_cast<unsigned short int>(vole_pixel[1] * scaling);
			magick_idx->blue  = static_cast<unsigned short int>(vole_pixel[2] * scaling);
		}
	}

	// changes are merged into the image
	mag.syncPixels();
}

template <typename T>
void VoleImage<T>::dumpBlob(std::string filename) {
	std::cout << "writing " << filename << std::endl;
	std::ofstream myfile;
	myfile.open (filename.c_str());
	int idx;
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			idx = y*maxX + x;
			T &vole_pixel = *(array + idx);
			myfile << "[" << vole_pixel[0] << "," << vole_pixel[1] << "," << vole_pixel[2] << "]" << std::endl;
		}
		myfile << std::endl;
	}
	myfile.close();
	std::cout << "finished " << filename << std::endl;
}


template <typename T>
void VoleImage<T>::fillInactivePixels(T &color) {
	T * current;
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			current = array + y*maxX + x;
			if (current->isInactive()) {
				current->set(0, color[0]);
				current->set(1, color[1]);
				current->set(2, color[2]);
			}
		}
	}
}

template <typename T>
void VoleImage<T>::copyActivePixels(VoleImage<T> &source) {
	// check if the dimensions are the same
	assert(maxY == source.getMaxY());
	assert(maxX == source.getMaxX());
	assert(array != 0);

	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			if (source(y, x).isActive()) {
				(array + y*maxX + x)->setActive();
			} else {
				(array + y*maxX + x)->setInactive();
			}
		}
	}
}

template <typename T>
void VoleImage<T>::magickShow() {
	Magick::Geometry geom(maxX, maxY);
	Magick::Image image(geom, "Black");
	voleImageToMagick(*this, image);
	image.display();
}

template <typename T>
VoleImage<T> &VoleImage<T>::operator=(VoleImage<T> &src) {
	deallocate();
	allocate(src.getMaxY(), src.getMaxX());

	std::memcpy(array, src.getBlob(), blobSize * sizeof(T));

	this->maxX = src.getMaxX();
	this->maxY = src.getMaxY();
	this->blobSize = src.getBlobSize();

	return *this;
}

template <typename T>
void VoleImage<T>::readImage(std::string filename) {
	deallocate();
	readImageWithoutCleaning(filename);
}


template <typename T>
void VoleImage<T>::allocate(int maxY, int maxX) {
	if (array != 0) { deallocate(); }
	
	blobSize = maxY * maxX;
	this->maxX = maxX;
	this->maxY = maxY;
	array = new T[blobSize];
//	array = static_cast<T *>(malloc(blobSize));
//	if (array == NULL) {
//		throw new std::bad_alloc();
//	}
}


template <typename T>
void VoleImage<T>::deallocate() {
	if (array != 0) {
		delete[] array;
		array = 0;
	}
//	free(array);
}


template <typename T>
void VoleImage<T>::readImageWithoutCleaning(std::string filename) {

	Magick::Image image(filename);
	image.modifyImage();
	maxY = image.rows();
	maxX = image.columns();

	allocate(maxY, maxX);

	magickToVoleImage(image, *this);
}


template <typename T>
void VoleImage<T>::grayScale(VoleImage<T> &dest) {
    Magick::Image img(Magick::Geometry(maxX, maxY), "Black");
    voleImageToMagick(*this, img);
    img.type(Magick::GrayscaleType);
    magickToVoleImage(img, dest);
}


template <typename T>
void VoleImage<T>::setZero() {
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			array[y*maxX + x].setZero();
		}
	}
}

template <typename T>
void VoleImage<T>::setZeroAndInactive() {
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			array[y*maxX + x].setZeroAndInactive();
		}
	}
}

#endif // VOLEIMAGE_CXX
