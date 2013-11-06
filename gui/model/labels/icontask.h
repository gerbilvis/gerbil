#ifndef ICONTASK_H
#define ICONTASK_H

#include <QObject>
#include <QThread>
#include <QColor>
#include <QVector>
#include <QSize>
#include <QImage>

#include <opencv2/core/core.hpp>

#include <boost/shared_ptr.hpp>
#include <tbb/task_group.h>

// Context struct for label icon computation.
// Bad style, but better than fiddling all parameters through constructors.
struct IconTaskCtx {
	explicit IconTaskCtx(int nlabels, const cv::Mat1s& full_labels,
						 const QSize& iconSize, const QVector<QColor>& colors)
			: nlabels(nlabels), full_labels(full_labels),
			  iconSize(iconSize), colors(colors)
	{}

	// default copy constructor and assignment

	// Inputs:
	// number of labels (including background == colors.size)
	const int nlabels;
	const cv::Mat1s full_labels;
	const QSize iconSize;
	const QVector<QColor> colors;

	// Result:
	QVector<QImage> icons;

};

typedef boost::shared_ptr<IconTaskCtx> IconTaskCtxPtr;

/** Computes label icons parallelized from input in IconTaskCtx. */
class IconTask : public QThread
{
	Q_OBJECT
public:
	explicit IconTask(IconTaskCtxPtr& ctxp, QObject *parent = 0);
public slots:
	/** Abort the computation.
	 *
	 * WARNING: Don't call this directly, always signal with queued
	 *          connection. abort() must execute in the IconTask thread!
	 */
	void abort();
protected:
	virtual void run();
signals:
	/** Label icons computed successfully. */
	void labelIconsComputed(const QVector<QImage>& icons);
	/** Task was aborted. */
	void taskAborted();
private:
	IconTaskCtxPtr ctxp;
	volatile bool abortFlag;
    tbb::task_group_context tbbTaskGroupContext;
};

#endif // ICONTASK_H
