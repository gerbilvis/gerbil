#ifndef PROGRESS_OBSERVER_H
#define PROGRESS_OBSERVER_H

namespace vole {

class ProgressObserver
{
public:
	virtual ~ProgressObserver() {}

	/// if false: cancel job
	virtual bool update(int percent) = 0;
};

/* For smaller tasks in a bigger calculation */
class ChainedProgressObserver : public ProgressObserver
{
public:
	ChainedProgressObserver(ProgressObserver *target, float fraction)
		: target(target), fraction(fraction) {}

	bool update(int percent) {
		return (!target) || target->update(percent * fraction);
	}

	ProgressObserver *target;
	float fraction;
};

}

#endif // PROGRESS_OBSERVER_H
