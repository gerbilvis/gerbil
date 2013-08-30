#ifndef DATARANGECUDA_H
#define DATARANGECUDA_H

#include "shared_data.h"
#include "background_task/background_task.h"
#include <tbb/task_group.h>

class DataRangeCuda : public BackgroundTask {
public:
	DataRangeCuda(SharedMultiImgPtr multi, SharedMultiImgRangePtr range,
		cv::Rect targetRoi = cv::Rect())
		: BackgroundTask(targetRoi), multi(multi), range(range) {}
	virtual ~DataRangeCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	SharedMultiImgRangePtr range;
};

#endif // DATARANGECUDA_H
