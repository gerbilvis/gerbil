#include <shared_data.h>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"
#include <background_task/background_task.h>
#include "clamptbb.h"

#define STOPWATCH_PRINT(stopwatch, message)

bool ClampTbb::run()
{
	multi_img *source = &**image;
	assert(0 != source);

	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = (*minmax)->minval;
	target->maxval = (*minmax)->maxval;

	vole::Stopwatch s;

	Clamp computeClamp(*source, *target);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()),
		computeClamp, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "Clamp TBB")

	if (includecache) {
		RebuildPixels rebuildPixels(*target);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()),
			rebuildPixels, tbb::auto_partitioner(), stopper);
		target->dirty.setTo(0);
		target->anydirt = false;
	} else {
		target->resetPixels();
	}

	if (stopper.is_group_execution_cancelled()) {
		delete target;
		return false;
	} else {
		SharedDataSwapLock lock(image->mutex);
		image->replace(target);
		return true;
	}
}
