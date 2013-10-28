#include <QPixmap>
#include <QImage>
#include <QPainter>

#include <opencv2/imgproc/imgproc.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>


#include <qtopencv.h>

#include "icontask.h"

#include <gerbil_gui_debug.h>

// TBB functor struct
struct ComputeIconMasks {

	ComputeIconMasks(IconTaskCtx& ctx, volatile bool& abortFlag)
		: ctx(ctx), abortFlag(abortFlag)
	{}

	void operator()(const tbb::blocked_range<short>& range) const {
		for( short label=range.begin(); label!=range.end(); ++label ) {
			if(abortFlag) {
				tbb::task::self().cancel_group_execution();
				return;
			}
			if(tbb::task::self().is_cancelled())
				return;

			// Compute mask.
			// For big images it might make sense to parallelize this on a
			// smaller granularity (pixel ranges).
			// And it might be a good idea to cache these.
			cv::Mat1b mask = (ctx.full_labels == label);

			if(tbb::task::self().is_cancelled())
				return;

			// convert QSize
			cv::Size iconSizecv(ctx.iconSize.width(), ctx.iconSize.height());
			// interpolate the mask to icon size
			cv::resize(mask, mask, iconSizecv, 0, 0, cv::INTER_AREA);

			if(tbb::task::self().is_cancelled())
				return;
			// the rest is probably too fast to allow checking for cancellation.

			QColor color = ctx.colors.at(label);

			if(0==label) {
				color = Qt::black;
			}

			// convert QColor to RGBA
			QRgb qrgb = color.rgba();


//			GGDBGM((format("label %1% color QRgb is r%2% g%3% b%4% a%5%")
//					%label%qRed(qrgb)%qGreen(qrgb)%qBlue(qrgb)%qAlpha(qrgb))
//				   <<endl);

			// fill icon with solid color in ARGB format
			cv::Vec4b argb(0, qRed(qrgb), qGreen(qrgb), qBlue(qrgb));



			cv::Mat4b icon = cv::Mat4b(mask.rows, mask.cols, argb);

			cv::Mat1b zero = cv::Mat1b::zeros(mask.rows, mask.cols);

			// Make ARGB 'array' which is interleaved to a true ARGB image
			// using mixChannels.
			cv::Mat in[] = {mask, zero, zero, zero};
			// Copy only the alpha channel (0) of the in array into the
			// alpha channel (0) of the ARGB icon.
			const int mix[] = {0,0};
			// 4 input matrices, 1 dest, 1 mix-pair
			cv::mixChannels(in,4, &icon,1, mix,1);
			// convert the result to a QImage
			QImage qimage = vole::Mat2QImage(icon);

			// FIXME
			// draw a border: temporary fix until the icon view does this
			QPainter p(&qimage);
			p.setPen(color);
			p.drawRect(0,0,ctx.iconSize.width()-1,ctx.iconSize.height()-1);

			ctx.icons[label] = qimage;
		}
	}

	IconTaskCtx& ctx;
	volatile bool& abortFlag;
};

IconTask::IconTask(IconTaskCtxPtr &ctxp, QObject *parent)
	:QThread(parent), ctxp(ctxp), abortFlag(false)
	//, taskgctxp(NULL)
{
}

void IconTask::abort()
{
	// if no tasks are executing yet, remember not to start
	abortFlag = true;

	// tell executing tasks to abort
	tbb::task::self().context()->cancel_group_execution();
}

void IconTask::run()
{
	assert(NULL != ctxp);
	IconTaskCtx ctx = *ctxp;
	assert(ctx.nlabels > 0);
	assert(ctx.nlabels == ctx.colors.size());
	assert(ctx.colors.size() == ctx.nlabels);
	assert(ctx.icons.size() == 0); // we do the allocation
	assert(ctx.iconSize.width() > 0 && ctx.iconSize.width() <= 256);
	assert(ctx.iconSize.height() > 0 && ctx.iconSize.height() <= 256);

	// init result vector
	ctx.icons = QVector<QImage>(ctx.nlabels);

	// No idea if it is necessary to get the task_group_context of the self-task
	// or if parallel_for will use it anyway.
	// FIXME:
	// Also there may be a race-condition, if cancel_group_execution() is reset
	// upon calling parallel_for.
	tbb::task_group_context*  taskgctxp = tbb::task::self().context();
	tbb::auto_partitioner partitioner;

	if(abortFlag) {
		emit taskAborted();
		return;
	}

	tbb::parallel_for(tbb::blocked_range<short>(0,ctx.nlabels),
					  ComputeIconMasks(ctx,abortFlag),
					  partitioner,
					  *taskgctxp);

	if(!abortFlag) {
		emit labelIconsComputed(ctx.icons);
	} else {
		emit taskAborted();
	}
}

