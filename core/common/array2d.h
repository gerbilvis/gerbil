#ifndef ARRAY2D_H
#define ARRAY2D_H

#include <cassert>
#include <cstddef>

namespace vole {

template <typename ContentT>
class Array2D {

public:

	Array2D() { blobInitialized = false; }

	Array2D(int yDim, int xDim) {
		blobInitialized = false;
		alloc(yDim, xDim);
	}

	void dim(int *yDim, int *xDim) {
		if (yDim != NULL) *yDim = maxY;
		if (xDim != NULL) *xDim = maxX;
	}

	ContentT *operator[](const unsigned int y) const { return blob[y]; }
	ContentT *operator[](unsigned int y) { return blob[y]; }

	ContentT &operator()(const unsigned int y, const unsigned int x) {
		return blob[y][x];
	}

	ContentT **operator*() {
		return blob;
	}

	void set(const unsigned int y, const unsigned int x, ContentT value) {
		blob[y][x] = value;
	}

	~Array2D() {
		dealloc();
	}


protected:

	void alloc(int xDim, int yDim) {
		if (blobInitialized == true) dealloc();
		maxX = xDim;
		maxY = yDim;
		blob = new ContentT*[maxY];
		for (int i = 0; i < yDim; ++i) {
			blob[i] = new ContentT[maxX];
		}
		blobInitialized = true;
	}

	void dealloc() {
		blobInitialized = false;
		for (int i = 0; i < maxY; i++) {
			delete[] blob[i];
		}
		delete[] blob;
		maxX = maxY = 0;
	}

int maxX, maxY;
bool blobInitialized;
ContentT **blob;

};
}

#endif // ARRAY2D_H
