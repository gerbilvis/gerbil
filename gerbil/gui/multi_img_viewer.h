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
#include <QMenu>
#include <cv.h>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);

	const multi_img* getImage() { return image; }
	Viewport* getViewport() { return viewport; }
	const multi_img::Mask& getMask() { return maskholder; }

	cv::Mat1s labels;

public slots:
	void rebuild(int bins = 0);
	void updateMask(int dim);
	void setImage(const multi_img *image, representation type);
	void setIlluminant(const std::vector<multi_img::Value> *, bool for_real);
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

protected:
	class BinsTbb : public BackgroundTask {
	public:
		BinsTbb(multi_img_ptr multi, label_ptr labels, 
			QVector<QColor> colors, bool ignoreLabels, 
			std::vector<multi_img::Value> illuminant, 
			ViewportCtx args, vpctx_ptr context, 
			sets_ptr current, sets_ptr temp, 
			std::vector<cv::Rect> sub, std::vector<cv::Rect> add, bool apply,
			cv::Rect targetRoi = cv::Rect(0, 0, 0, 0)) 
			: BackgroundTask(targetRoi), multi(multi), labels(labels), colors(colors),
			ignoreLabels(ignoreLabels), illuminant(illuminant), args(args),
			context(context), current(current), temp(temp), sub(sub), add(add), apply(apply) {}
		virtual ~BinsTbb() {}
		virtual bool run();
		virtual void cancel() { stopper.cancel_group_execution(); }
	protected:
		tbb::task_group_context stopper;

		class Accumulate {
		public:
			Accumulate(bool substract, multi_img &multi, cv::Mat1s &labels, 
				int nbins, multi_img::Value binsize, bool ignoreLabels,
				std::vector<multi_img::Value> &illuminant, 
				std::vector<BinSet> &sets, cv::Rect region) 
				: substract(substract), multi(multi), labels(labels), nbins(nbins), binsize(binsize), 
				illuminant(illuminant), ignoreLabels(ignoreLabels), sets(sets), region(region) {}
			void operator()(const tbb::blocked_range2d<int> &r) const;
		private:
			bool substract;
			multi_img &multi;
			cv::Mat1s &labels;
			int nbins;
			multi_img::Value binsize;
			bool ignoreLabels;
			std::vector<multi_img::Value> &illuminant;
			std::vector<BinSet> &sets;
			cv::Rect region;
		};
		
		multi_img_ptr multi;
		label_ptr labels;
		QVector<QColor> colors;
		bool ignoreLabels;
		std::vector<multi_img::Value> illuminant;
		ViewportCtx args;

		vpctx_ptr context;
		sets_ptr current;
		sets_ptr temp;
		
		std::vector<cv::Rect> sub;
		std::vector<cv::Rect> add;
		bool apply;
	};

	/* translate image value to value in our coordinate system */
	inline multi_img::Value curpos(multi_img::Value val, int dim) {
		multi_img::Value curpos = (val - image->minval) / binsize;
		if (illuminant)
			curpos /= (*illuminant)[dim];
		return curpos;
	}

    void changeEvent(QEvent *e);
	void createBins();

	// helpers for createMask
	void fillMaskSingle(int dim, int sel);
	void fillMaskLimiters(const std::vector<std::pair<int, int> >& limits);
	void updateMaskLimiters(const std::vector<std::pair<int, int> >&, int dim);

	const multi_img *image;
	const std::vector<multi_img::Value> *illuminant;
	bool ignoreLabels;
	multi_img::Mask maskholder;
	bool maskValid;

private:
	void createLimiterMenu();

	// current number of bins shown
	int nbins;
	// respective data range of each bin
	multi_img::Value binsize;
	QMenu limiterMenu;
	QVector<QColor> labelColors;
};

#endif // MULTI_IMG_VIEWER_H
