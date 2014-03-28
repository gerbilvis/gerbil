#ifndef CLAMPCUDA_H
#define CLAMPCUDA_H

#include "shared_data.h"
#include "background_task/background_task.h"
#include <tbb/task_group.h>

class ClampCuda : public BackgroundTask {
public:
	ClampCuda(SharedMultiImgPtr image, SharedMultiImgPtr minmax,
		bool includecache = true)
		: BackgroundTask(), image(image), minmax(minmax), includecache(includecache)
	{}
	virtual ~ClampCuda() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr image;
	SharedMultiImgPtr minmax;
	bool includecache;
};
#endif // CLAMPCUDA_H
