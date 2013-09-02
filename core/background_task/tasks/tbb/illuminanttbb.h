#ifndef ILLUMINANTTBB_H
#define ILLUMINANTTBB_H

class IlluminantTbb : public BackgroundTask {
public:
	IlluminantTbb(SharedMultiImgPtr multi, const Illuminant& il, bool remove,
		cv::Rect targetRoi = cv::Rect(), bool includecache = true)
		: BackgroundTask(targetRoi), multi(multi),
		il(il), remove(remove), includecache(includecache) {}
	virtual ~IlluminantTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	Illuminant il;
	bool remove;
	bool includecache;
};
#endif // ILLUMINANTTBB_H
