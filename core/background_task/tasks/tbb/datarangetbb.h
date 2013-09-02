#ifndef DATARANGETBB_H
#define DATARANGETBB_H

class DataRangeTbb : public BackgroundTask {
public:
	DataRangeTbb(SharedMultiImgPtr multi, SharedMultiImgRangePtr range,
		cv::Rect targetRoi = cv::Rect())
		: BackgroundTask(targetRoi), multi(multi), range(range) {}
	virtual ~DataRangeTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;

	SharedMultiImgPtr multi;
	SharedMultiImgRangePtr range;
};
#endif // DATARANGETBB_H
