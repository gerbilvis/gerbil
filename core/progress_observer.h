#ifndef PROGRESS_OBSERVER_H
#define PROGRESS_OBSERVER_H

class ProgressObserver
{
public:
	virtual ~ProgressObserver() {}

	/** provide an update and check if you should continue
	 * @param report the progress to report, between 0 and 1 (100% done)
	 * @param incremental the report is an increment (false: absolute progress)
	 * @return if false, cancel the computation ASAP
	 * @note This is not thread-safe. But progress output is not
	 *  mission-critical. So it is o.k. to call from different threads.
	 *  The abort mechanism is thread-safe though.
	 */
	virtual bool update(float report, bool incremental = false) = 0;
};

/* For smaller tasks in a bigger calculation */
class ChainedProgressObserver : public ProgressObserver
{
public:
	ChainedProgressObserver(ProgressObserver *target, float fraction)
		: target(target), fraction(fraction), reported(0.f) {}

	bool update(float report, bool incremental = false) {
		if (!target)
			return true;
		/* Note: we need to do our own book-keeping to translate to correct
		 * incremental updates */
		if (incremental) {
			reported += report;
			return target->update(report * fraction, incremental);
		} else {
			float gain = report - reported;
			reported = report;
			return target->update(gain * fraction, true);
		}
	}

	ProgressObserver *target;
	float fraction;
	float reported;
};

#endif // PROGRESS_OBSERVER_H
