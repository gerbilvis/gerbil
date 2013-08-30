#ifndef PCATBB_H
#define PCATBB_H

class PcaTbb : public BackgroundTask {
public:
	PcaTbb(SharedMultiImgPtr source, SharedMultiImgPtr current, unsigned int components = 0,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), source(source), current(current),
		components(components), includecache(includecache) {}
	virtual ~PcaTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	unsigned int components;
	bool includecache;
};

#endif // PCATBB_H
