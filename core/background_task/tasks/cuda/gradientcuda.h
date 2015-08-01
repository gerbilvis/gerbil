#ifndef GRADIENTCUDA_H
#define GRADIENTCUDA_H

#ifdef GERBIL_CUDA

#include "shared_data.h"
#include "background_task/background_task.h"
#include <tbb/task_group.h>

class GradientCuda : public BackgroundTask {
public:
	GradientCuda(SharedMultiImgPtr source,
				 SharedMultiImgPtr current,
				 bool includecache = true)
		: BackgroundTask(),
		  source(source),
		  current(current),
		  includecache(includecache)
	{}
	virtual ~GradientCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	bool includecache;
};

#endif

#endif // GRADIENTCUDA_H
