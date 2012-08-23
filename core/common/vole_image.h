/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_IMAGE_H
#define VOLE_IMAGE_H

#include <iostream>
#include <fstream>
// #include <Magick++.h>

template <typename T>
class VoleImage {

public:
	VoleImage() { maxX = 0; maxY = 0; array = 0; }
	VoleImage(std::string the_filename);
	VoleImage(int height, int width);
	VoleImage(const VoleImage<T> &copy);
	~VoleImage();

	int getNChannels() { if (array != 0) return array[0].getDim(); }


	//! free()s array and reads a new image
	void readImage(std::string filename);

	//! saves array to filename
	void writeImage(std::string filename);
	void writePPM(std::string filename, bool raw);

	int getMaxX() const { return maxX; }
	int getMaxY() const { return maxY; }

	//! returns the contents
	T &operator()(int y, int x);

	//! Fills region with a subregion of the image, limited by  minY, minX (inclusively) and maxY, maxX (exclusively);  
	void getRegion(VoleImage<T> &region, int minY, int minX, int maxY, int maxX);

	//! returns the sum of all channels in position [y, x]
	//T operator()(int y, int x);

	T *getBlob() const { return array; }
	int getBlobSize() const { return blobSize; }
	std::string getFilename() { return filename; }

	void allocate(int maxY, int maxX);
	void deallocate();

	void magickToVoleImage(Magick::Image &mag, VoleImage<T> &img);
	void voleImageToMagick(VoleImage<T> &img, Magick::Image &mag);
	void magickShow();
	void dumpBlob(std::string filename);
	void fillInactivePixels(T &color);
	void copyActivePixels(VoleImage<T> &source);

	void grayScale(VoleImage<T> &dest); 
	void setZero();
	void setZeroAndInactive();

	VoleImage<T> &operator=(VoleImage<T> &src);

protected:
	/// breite des bildes
	int maxX;
	/// hoehe des bildes
	int maxY;
	/// groesse des Pixelspeichers
	int blobSize;
	/// pixelspeicher
	T *array;
	std::string filename;

	//! does not free() array before the image is read; necessary for the constructor
	void readImageWithoutCleaning(std::string filename);
};

#endif // VOLE_IMAGE_H
