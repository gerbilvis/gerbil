#ifndef BGRTBB_H
#define BGRTBB_H

#include "shared_data.h"
#include "background_task/background_task.h"
#include <tbb/task_group.h>

class BgrTbb : public BackgroundTask {
public:
	BgrTbb(SharedMultiImgPtr multi, mat3f_ptr bgr,
		cv::Rect targetRoi = cv::Rect())
		: BackgroundTask(targetRoi), multi(multi), bgr(bgr) {}
	virtual ~BgrTbb() {}
	virtual bool run();

	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr multi;
	mat3f_ptr bgr;
};



#endif // BGRTBB_H
