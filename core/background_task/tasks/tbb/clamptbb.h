#ifndef CLAMPTBB_H
#define CLAMPTBB_H

class ClampTbb : public BackgroundTask {
public:
	ClampTbb(SharedMultiImgPtr image, SharedMultiImgPtr minmax, bool includecache = true)
		: BackgroundTask(), image(image),	minmax(minmax),
		  includecache(includecache)
	{}
	virtual ~ClampTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;


	SharedMultiImgPtr image;

	SharedMultiImgPtr minmax;
	bool includecache;
};

#endif // CLAMPTBB_H
