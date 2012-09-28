#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "multi_img.h"

#include <shared_data.h>
#include <background_task.h>

#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>

#include <vector>
#include <map>
#include <QMenu>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);

	multi_img_ptr getImage() { return image; }
	Viewport* getViewport() { return viewport; }
	const multi_img::Mask& getMask() { return maskholder; }
	int getSelection() { return viewport->selection; }
	representation getType() { SharedDataHold l(viewport->ctx->lock); return (*viewport->ctx)->type; }
	void enableBinSlider(bool enable) { binSlider->setEnabled(enable); }

	cv::Mat1s labels;

public slots:
	void updateMask(int dim);
	void subPixels(const std::map<std::pair<int, int>, short> &points);
	void addPixels(const std::map<std::pair<int, int>, short> &points);
	void subImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void addImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(multi_img_ptr image, representation type, cv::Rect roi);
	void setIlluminant(const std::vector<multi_img::Value> &, bool for_real);
	void changeBinCount(int bins);
	void updateBinning(int bins);
	void finishBinCountChange(bool success);
	void updateLabels();
	void toggleFold();
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void toggleLimiters(bool toggle);
	void setAlpha(int);
	void overlay(int x, int y);
	void showLimiterMenu();
	void setActive()	{ viewport->active = true; viewport->update(); }
	void setInactive()	{ viewport->active = false; viewport->update(); }
	void updateLabelColors(const QVector<QColor> &labelColors, bool changed);

signals:
	void newOverlay();
	void folding();
	void setGUIEnabled(bool enable, TaskType tt);
	void finishTask(bool success);

protected:
	class BinsTbb : public BackgroundTask {
	public:
		BinsTbb(multi_img_ptr multi, const cv::Mat1s &labels, 
			const QVector<QColor> &colors,
			const std::vector<multi_img::Value> &illuminant, 
			const ViewportCtx &args, vpctx_ptr context, 
			sets_ptr current, sets_ptr temp = sets_ptr(new SharedData<std::vector<BinSet> >(NULL)), 
			const std::vector<cv::Rect> &sub = std::vector<cv::Rect>(),
			const std::vector<cv::Rect> &add = std::vector<cv::Rect>(), 
			bool inplace = false, bool apply = true, cv::Rect targetRoi = cv::Rect(0, 0, 0, 0)) 
			: BackgroundTask(targetRoi), multi(multi), labels(labels), colors(colors),
			illuminant(illuminant), args(args), context(context), 
			current(current), temp(temp), sub(sub), add(add), inplace(inplace), apply(apply) {}
		virtual ~BinsTbb() {}
		virtual bool run();
		virtual void cancel() { stopper.cancel_group_execution(); }
	protected:
		tbb::task_group_context stopper;

		class Accumulate {
		public:
			Accumulate(bool substract, multi_img &multi, cv::Mat1s &labels, 
				int nbins, multi_img::Value binsize, multi_img::Value minval, bool ignoreLabels,
				std::vector<multi_img::Value> &illuminant, 
				std::vector<BinSet> &sets) 
				: substract(substract), multi(multi), labels(labels), nbins(nbins), binsize(binsize),
				minval(minval), illuminant(illuminant), ignoreLabels(ignoreLabels), sets(sets) {}
			void operator()(const tbb::blocked_range2d<int> &r) const;
		private:
			bool substract;
			multi_img &multi;
			cv::Mat1s &labels;
			int nbins;
			multi_img::Value binsize;
			multi_img::Value minval;
			bool ignoreLabels;
			std::vector<multi_img::Value> &illuminant;
			std::vector<BinSet> &sets;
		};
		
		multi_img_ptr multi;
		cv::Mat1s labels;
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

	/* translate image value to value in our coordinate system */
	inline multi_img::Value curpos(multi_img::Value val, int dim) {
		SharedDataHold ctxlock(viewport->ctx->lock);
		multi_img::Value curpos = 
			(val - (*viewport->ctx)->minval) / (*viewport->ctx)->binsize;
		if (!illuminant.empty())
			curpos /= illuminant[dim];
		return curpos;
	}

    void changeEvent(QEvent *e);

	// helpers for createMask
	void fillMaskSingle(int dim, int sel);
	void fillMaskLimiters(const std::vector<std::pair<int, int> >& limits);
	void updateMaskLimiters(const std::vector<std::pair<int, int> >&, int dim);
	void setTitle(representation type, multi_img::Value min, multi_img::Value max);

	multi_img_ptr image;
	std::vector<multi_img::Value> illuminant;
	bool ignoreLabels;
	multi_img::Mask maskholder;
	bool maskValid;
	bool maskReset;
	bool titleReset;

protected slots:
	void render(bool necessary = true);

private:
	void createLimiterMenu();

	// respective data range of each bin
	QMenu limiterMenu;
	QVector<QColor> labelColors;
};

#endif // MULTI_IMG_VIEWER_H
