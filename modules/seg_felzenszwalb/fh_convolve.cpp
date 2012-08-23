#include "fh_convolve.h"
#include <algorithm>
#include <cmath>

namespace vole {

void convolve_even(image<float> *src, image<float> *dst, 
			  std::vector<float> &mask) {
  int width = src->width();
  int height = src->height();
  int len = mask.size();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = mask[0] * imRef(src, x, y);
      for (int i = 1; i < len; i++) {
	sum += mask[i] * 
	  (imRef(src, std::max(x-i,0), y) + 
	   imRef(src, std::min(x+i, width-1), y));
      }
      imRef(dst, y, x) = sum;
    }
  }
}

/* convolve src with mask.  dst is flipped! */
void convolve_odd(image<float> *src, image<float> *dst, 
			 std::vector<float> &mask) {
  int width = src->width();
  int height = src->height();
  int len = mask.size();

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = mask[0] * imRef(src, x, y);
      for (int i = 1; i < len; i++) {
	sum += mask[i] * 
	  (imRef(src, std::max(x-i,0), y) - 
	   imRef(src, std::min(x+i, width-1), y));
      }
      imRef(dst, y, x) = sum;
    }
  }
}

}
