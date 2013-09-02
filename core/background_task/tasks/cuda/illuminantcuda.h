#ifndef ILLUMINANTCUDA_H
#define ILLUMINANTCUDA_H

#include "shared_data.h"
#include "background_task/background_task.h"
#include <tbb/task_group.h>

#include "multi_img/illuminant.h"

class IlluminantCuda : public BackgroundTask {
public:
	IlluminantCuda(SharedMultiImgPtr multi, const Illuminant& il, bool remove,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), multi(multi),
		il(il), remove(remove), includecache(includecache) {}
	virtual ~IlluminantCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	Illuminant il;
	bool remove;
	bool includecache;
};
#endif // ILLUMINANTCUDA_H
