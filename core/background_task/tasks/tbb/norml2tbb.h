#ifndef NORML2TBB_H
#define NORML2TBB_H

class NormL2Tbb : public BackgroundTask {
public:
	NormL2Tbb(SharedMultiImgPtr source, SharedMultiImgPtr current)
		: BackgroundTask(), source(source), current(current) {}
	virtual ~NormL2Tbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
};

#endif // NORML2TBB_H
