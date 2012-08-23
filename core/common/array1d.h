#ifndef ARRAY1D_H
#define ARRAY1D_H

#include <cassert>


template <typename ContentT>
class Array1D {

public:

	Array1D(int xDim) {
		maxX = xDim;

		blob = new ContentT[maxX];
	}

	void dim(int *xDim) {
		*xDim = maxX;
	}

	ContentT &operator()(const unsigned int x) {
		return blob[x];
	}

	ContentT *operator*() {
		return blob;
	}

	void set(const unsigned int x, ContentT value) {
		blob[x] = value;
	}

	~Array1D() {
		delete[] blob;
	}


protected:

int maxX;
ContentT *blob;

};


#endif // ARRAY1D_H
