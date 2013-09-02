
#include <shared_data.h>

#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_reduce.h>

#include <stopwatch.h>

#include "multi_img/multi_img_tbb.h"
#include <background_task/background_task.h>
#include "datarangetbb.h"

#define STOPWATCH_PRINT(stopwatch, message)


bool DataRangeTbb::run()
{
	vole::Stopwatch s;

	DetermineRange determineRange(**multi);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, (*multi)->size()),
		determineRange, tbb::auto_partitioner(), stopper);

	STOPWATCH_PRINT(s, "DataRange TBB")

	if (!stopper.is_group_execution_cancelled()) {
		SharedDataSwapLock lock(range->mutex);
		(*range)->min = determineRange.GetMin();
		(*range)->max = determineRange.GetMax();
		return true;
	} else {
		return false;
	}
}
