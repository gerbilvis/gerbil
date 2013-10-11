#include <shared_data.h>

#include <tbb/task_group.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <stopwatch.h>

#include <multi_img/illuminant.h>
#include "multi_img/multi_img_tbb.h"

#include <background_task/background_task.h>
#include "illuminanttbb.h"

#define STOPWATCH_PRINT(stopwatch, message)

bool IlluminantTbb::run()
{
	multi_img *source = &**multi;
	assert(0 != source);
	multi_img *target = new multi_img(source->height, source->width, source->size());
	target->roi = source->roi;
	target->meta = source->meta;
	target->minval = source->minval;
	target->maxval = source->maxval;

	vole::Stopwatch s;

	Illumination computeIllumination(*source, *target, il, remove);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, target->size()),
		computeIllumination, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "Illuminant TBB")

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
		SharedDataSwapLock lock(multi->mutex);
		multi->replace(target);
		return true;
	}
}
