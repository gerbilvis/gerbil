#include <multi_img_tasks.h>
#include "rgbserial.h"

bool RgbSerial::run()
{
	if (!MultiImg::BgrSerial::run())
		return false;
	cv::Mat3f &bgrmat = *(*bgr);
	QImage *newRgb = new QImage(bgrmat.cols, bgrmat.rows, QImage::Format_ARGB32);
	for (int y = 0; y < bgrmat.rows; ++y) {
		cv::Vec3f *row = bgrmat[y];
		QRgb *destrow = (QRgb*)newRgb->scanLine(y);
		for (int x = 0; x < bgrmat.cols; ++x) {
			cv::Vec3f &c = row[x];
			destrow[x] = qRgb(c[2]*255., c[1]*255., c[0]*255.);
		}
	}
	SharedDataSwapLock lock(rgb->mutex);
	delete rgb->swap(newRgb);
	return true;
}
