/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/* image conversion */

#ifndef VOLE_CONV_H
#define VOLE_CONV_H

#include "image.h"
#include "imutil.h"
#include "misc.h"

#define	RED_WEIGHT	0.299
#define GREEN_WEIGHT	0.587
#define BLUE_WEIGHT	0.114

namespace vole {

	image<uchar> *imageRGBtoGRAY(image<rgb> *input);
	image<rgb>   *imageGRAYtoRGB(image<uchar> *input);
	image<float> *imageUCHARtoFLOAT(image<uchar> *input);
	image<float> *imageINTtoFLOAT(image<int> *input);
	image<uchar> *imageFLOATtoUCHAR(image<float> *input, float min, float max);
	image<uchar> *imageFLOATtoUCHAR(image<float> *input);
	image<long>  *imageUCHARtoLONG(image<uchar> *input);
	image<uchar> *imageLONGtoUCHAR(image<long> *input, long min, long max);
	image<uchar> *imageLONGtoUCHAR(image<long> *input);
	image<uchar> *imageSHORTtoUCHAR(image<short> *input, short min, short max);
	image<uchar> *imageSHORTtoUCHAR(image<short> *input);

}

#endif
