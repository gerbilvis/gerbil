#ifndef GRADIENTTBB_H
#define GRADIENTTBB_H

class GradientTbb : public BackgroundTask {
public:
	GradientTbb(SharedMultiImgPtr source, SharedMultiImgPtr current,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), source(source),
		current(current), includecache(includecache) {}
	virtual ~GradientTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr source;
	SharedMultiImgPtr current;
	bool includecache;
};

#endif // GRADIENTTBB_H
