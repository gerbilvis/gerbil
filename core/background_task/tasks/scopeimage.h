#ifndef SCOPEIMAGE_H
#define SCOPEIMAGE_H

class ScopeImage : public BackgroundTask {
public:
	ScopeImage(SharedMultiImgPtr full, SharedMultiImgPtr scoped, cv::Rect roi)
		: BackgroundTask(roi), full(full), scoped(scoped) {}
	virtual ~ScopeImage() {}
	virtual bool run() {
		// using SharedData<multi_img_base>::getBase() to get multi_img_base object
		multi_img *target =  new multi_img(full->getBase(), targetRoi);
		SharedDataSwapLock lock(scoped->mutex);
		scoped->replace(target);
		return true;
	}
	virtual void cancel() {}
protected:
	SharedMultiImgPtr full;
	SharedMultiImgPtr scoped;
};

#endif // SCOPEIMAGE_H
