#ifndef BAND2QIMAGETBB_H
#define BAND2QIMAGETBB_H

#include <tbb/task_group.h>

class Band2QImageTbb : public BackgroundTask {
public:
	Band2QImageTbb(SharedMultiImgPtr multi, qimage_ptr image, size_t band)
		: BackgroundTask(), multi(multi), image(image), band(band) {}
	virtual ~Band2QImageTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	qimage_ptr image;
	size_t band;
};

#endif // BAND2QIMAGETBB_H
