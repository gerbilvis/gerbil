#ifndef BGRTBB_H
#define BGRTBB_H

#include <multi_img/multi_img_tbb.h>
#include <multi_img/cieobserver.h>

class BgrTbb : public BackgroundTask {
public:
	BgrTbb(SharedMultiImgPtr multi, mat3f_ptr bgr,
		cv::Rect targetRoi = cv::Rect())
		: BackgroundTask(targetRoi), multi(multi), bgr(bgr) {}
	virtual ~BgrTbb() {}
	virtual bool run() {
		multi_img_base& source = multi->getBase();
		cv::Mat_<cv::Vec3f> xyz(source.height, source.width, 0.);
		float greensum = 0.f;
		for (size_t i = 0; i < source.size(); ++i) {
			int idx = ((int)(source.meta[i].center + 0.5f) - 360) / 5;
			if (idx < 0 || idx > 94)
				continue;

			multi_img::Band band;
			source.getBand(i, band);
			Xyz computeXyz(source, xyz, band, idx);
			tbb::parallel_for(tbb::blocked_range2d<int>(0, xyz.rows, 0, xyz.cols),
				computeXyz, tbb::auto_partitioner(), stopper);

			greensum += CIEObserver::y[idx];

			if (stopper.is_group_execution_cancelled())
				break;
		}

		if (greensum == 0.f)
			greensum = 1.f;

		cv::Mat_<cv::Vec3f> *newBgr = new cv::Mat_<cv::Vec3f>(source.height, source.width);
		Bgr computeBgr(source, xyz, *newBgr, greensum);
		tbb::parallel_for(tbb::blocked_range2d<int>(0, newBgr->rows, 0, newBgr->cols),
			computeBgr, tbb::auto_partitioner(), stopper);

		if (stopper.is_group_execution_cancelled()) {
			delete newBgr;
			return false;
		} else {
			SharedDataSwapLock lock(bgr->mutex);
			delete bgr->swap(newBgr);
			return true;
		}
	}

	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr multi;
	mat3f_ptr bgr;
};

#endif // BGRTBB_H
