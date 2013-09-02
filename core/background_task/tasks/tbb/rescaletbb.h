#ifndef RESCALETBB_H
#define RESCALETBB_H

class RescaleTbb : public BackgroundTask {
public:
	RescaleTbb(SharedMultiImgPtr source, SharedMultiImgPtr current, size_t newsize,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), source(source), current(current),
		newsize(newsize), includecache(includecache) {}
	virtual ~RescaleTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;
	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	size_t newsize;
	bool includecache;
};

#endif // RESCALETBB_H
