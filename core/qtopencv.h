/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef QTOPENCV_H
#define QTOPENCV_H

#ifdef WITH_OPENCV2
#ifdef WITH_QT

#include "multi_img.h"
#include <opencv2/core/core.hpp>
#include <QImage>
#include <QVector>
#include <QColor>

namespace vole {

/**
	Helper functions for interoperation between Qt and OpenCV.
**/

cv::Mat3b QImage2Mat(const QImage &src);
QImage Mat2QImage(const cv::Mat3b &src);
// untested: return ARGB32 image with black background and provided color as foreground
QImage Mask2QImage(const cv::Mat1b &src, const QColor &color);
QImage Mat2QImage(const cv::Mat_<double> &src);
QImage Band2QImage(const multi_img::Band src, multi_img::Value minval, multi_img::Value maxval);

/** Convert ARGB cv::Mat to QImage. */
QImage Mat2QImage(const cv::Mat4b &src);

inline cv::Vec3b QColor2Vec(const QColor &src)
{
	return cv::Vec3b(src.blue(), src.green(), src.red());
}

inline std::vector<cv::Vec3b> QColor2Vec(const QVector<QColor> & src)
{
	std::vector<cv::Vec3b> dest(src.size());
	for (int i = 0; i < src.size(); ++i)
		dest[i] = QColor2Vec(src[i]);
	return dest;
}

inline QColor Vec2QColor(const cv::Vec3b &src)
{
	return QColor(src[2], src[1], src[0]);
}

inline QVector<QColor> Vec2QColor(const std::vector<cv::Vec3b> & src)
{
	QVector<QColor> dest(src.size());
	for (size_t i = 0; i < src.size(); ++i)
		dest[i] = Vec2QColor(src[i]);
	return dest;
}

}

#endif
#endif
#endif // QTOPENCV_H
