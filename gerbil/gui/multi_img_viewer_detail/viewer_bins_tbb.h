#ifndef MULTI_IMG_VIEWER_BINS_TBB_H
#define MULTI_IMG_VIEWER_BINS_TBB_H

class ViewerBinsTbb : public BackgroundTask {
public:
	ViewerBinsTbb(
		SharedMultiImgPtr multi, const cv::Mat1s &labels,
		const QVector<QColor> &colors,
		const std::vector<multi_img::Value> &illuminant,
		const ViewportCtx &args, vpctx_ptr context,
		sets_ptr current, sets_ptr temp = sets_ptr(new SharedData<std::vector<BinSet> >(NULL)),
		const std::vector<cv::Rect> &sub = std::vector<cv::Rect>(),
		const std::vector<cv::Rect> &add = std::vector<cv::Rect>(),
		const cv::Mat1b &mask = cv::Mat1b(),
		bool inplace = false, bool apply = true, cv::Rect targetRoi = cv::Rect(0, 0, 0, 0))
		: BackgroundTask(targetRoi), multi(multi), labels(labels), colors(colors),
		illuminant(illuminant), args(args), context(context),
		current(current), temp(temp), sub(sub), add(add), mask(mask), inplace(inplace), apply(apply) {}
	virtual ~ViewerBinsTbb() {}
	virtual bool run();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;


	SharedMultiImgPtr multi;
	cv::Mat1s labels;
	cv::Mat1b mask;
	QVector<QColor> colors;
	std::vector<multi_img::Value> illuminant;
	ViewportCtx args;

	vpctx_ptr context;
	sets_ptr current;
	sets_ptr temp;

	std::vector<cv::Rect> sub;
	std::vector<cv::Rect> add;
	bool inplace;
	bool apply;
};
#endif // MULTI_IMG_VIEWER_BINS_TBB_H
