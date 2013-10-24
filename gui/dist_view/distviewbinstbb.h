#ifndef DISTVIEWBINSTBB_H
#define DISTVIEWBINSTBB_H

#include "../dist_view/compute.h"

#include <background_task/background_task.h>
#include <shared_data.h>

#include <QImage>
#include <QColor>
#include <QVector>

#include <tbb/task.h>
#include <tbb/blocked_range2d.h>

class DistviewBinsTbb : public BackgroundTask {
public:
	DistviewBinsTbb(
		SharedMultiImgPtr multi, const cv::Mat1s &labels,
		const QVector<QColor> &colors,
		const std::vector<multi_img::Value> &illuminant,
		const ViewportCtx &args, vpctx_ptr context,
		sets_ptr current,
		sets_ptr temp = sets_ptr(new SharedData<std::vector<BinSet> >(NULL)),
		const std::vector<cv::Rect> &sub = std::vector<cv::Rect>(),
		const std::vector<cv::Rect> &add = std::vector<cv::Rect>(),
		const cv::Mat1b &mask = cv::Mat1b(),
		bool inplace = false, bool apply = true, cv::Rect targetRoi = cv::Rect())
		: BackgroundTask(targetRoi), multi(multi), labels(labels), colors(colors),
		illuminant(illuminant), args(args), context(context),
		current(current), temp(temp), sub(sub), add(add), mask(mask), inplace(inplace), apply(apply) {}
	virtual ~DistviewBinsTbb() {}
	virtual bool run();
	// helper to run(): update viewport context
	void updateContext();
	virtual void cancel() { stopper.cancel_group_execution(); }
protected:
	tbb::task_group_context stopper;


	SharedMultiImgPtr multi;
	const cv::Mat1s labels;
	const cv::Mat1b mask;
	QVector<QColor> colors;
	std::vector<multi_img::Value> illuminant;
	// source context
	ViewportCtx args;

	// target context
	vpctx_ptr context;
	sets_ptr current;
	sets_ptr temp;

	std::vector<cv::Rect> sub;
	std::vector<cv::Rect> add;
	bool inplace;
	bool apply;
};

#endif // DISTVIEWBINSTBB_H
