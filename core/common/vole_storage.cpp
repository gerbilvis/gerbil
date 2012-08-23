/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "vole_storage.h"

namespace vole {


void Storage::write(const char *filename, CvMat *blob, const char *description) {
	CvFileStorage *storage = cvOpenFileStorage(filename, NULL, CV_STORAGE_WRITE);
	cvWrite(storage, description, blob);
	cvReleaseFileStorage(&storage);
}

void Storage::write(const char *filename, cv::Mat blob, const char *description) {
	CvMat blob2 = CvMat(blob);
	write(filename, &blob2, description);
}

cv::Mat Storage::read(const char *filename, const char *description) {

	CvFileStorage *storage = cvOpenFileStorage(filename, NULL, CV_STORAGE_READ);
	CvFileNode *good = cvGetFileNodeByName(storage, NULL, description);
	CvMat *blubb = static_cast<CvMat *>(cvRead(storage, good));

	bool copyContent = true;
	cv::Mat m(blubb, copyContent); 
	cvReleaseFileStorage(&storage);
	return m;
}

}
