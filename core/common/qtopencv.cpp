/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifdef WITH_OPENCV2
#ifdef WITH_QT

#include "qtopencv.h"

namespace vole {

cv::Mat3b QImage2Mat(const QImage &src) {
	unsigned int height = src.height();
	unsigned int width = src.width();

	cv::Mat3b dest(height, width);
	for (unsigned int y = 0; y < height; ++y) {
		cv::Vec3b *destrow = dest[y];
		for (unsigned int x = 0; x < width; ++x) {
			QRgb pxl = src.pixel(x, y);
			destrow[x] = cv::Vec3b(qBlue(pxl), qGreen(pxl), qRed(pxl));
		}
	}
	return dest;
}

QImage Mat2QImage(const cv::Mat3b &src) {
	QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
	for (int y = 0; y < src.rows; ++y) {
		const cv::Vec3b *srcrow = src[y];
		QRgb *destrow = (QRgb*)dest.scanLine(y);
		for (int x = 0; x < src.cols; ++x) {
			destrow[x] = qRgba(srcrow[x][2], srcrow[x][1], srcrow[x][0], 255);
		}
	}
	return dest;
}


QImage Mat2QImage(const cv::Mat_<double> &src)
{
	double scale = 255.0;
	QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
	for (int y = 0; y < src.rows; ++y) {
		const double *srcrow = src[y];
		QRgb *destrow = (QRgb*)dest.scanLine(y);
		for (int x = 0; x < src.cols; ++x) {
			unsigned int color = srcrow[x] * scale;
			destrow[x] = qRgba(color, color, color, 255);
		}
	}
	return dest;
}

QImage Band2QImage(const multi_img::Band src,
				   multi_img::Value minval, multi_img::Value maxval)
{
	multi_img::Value scale = 255.0/(maxval - minval);
	QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
	for (int y = 0; y < src.rows; ++y) {
		const multi_img::Value *srcrow = src[y];
		QRgb *destrow = (QRgb*)dest.scanLine(y);
		for (int x = 0; x < src.cols; ++x) {
			unsigned int color = (srcrow[x] - minval) * scale;
			destrow[x] = qRgba(color, color, color, 255);
		}
	}
	return dest;
}

}

#endif
#endif
