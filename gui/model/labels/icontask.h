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
#if !defined(Q_MOC_RUN)
#include <tbb/task_group.h>
#endif

// Context struct for label icon computation.
// Bad style, but better than fiddling all parameters through constructors.
struct IconTaskCtx {
	explicit IconTaskCtx(int nlabels,
						 const cv::Mat1s& full_labels,
						 const cv::Mat1s& roi_labels,
						 const QSize& iconSize,
						 bool applyROI,
						 const QVector<QColor>& colors)
			: nlabels(nlabels),
			  full_labels(full_labels),
			  roi_labels(roi_labels),
			  iconSize(iconSize),
			  applyROI(applyROI),
			  colors(colors)
	{}

	// default copy constructor and assignment

	// Inputs:
	// number of labels (including background == colors.size)
	const int nlabels;
	const cv::Mat1s full_labels;
	const cv::Mat1s roi_labels;
	const QSize iconSize;
	const bool applyROI;
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
	virtual ~IconTask();
	enum {IconSizeMin = 4};
	enum {IconSizeMax = 1024};
	QSize getIconSize() const { return ctxp->iconSize; }
	bool getApplyROI() const { return ctxp->applyROI; }

public slots:
	/** Abort the computation.
	 *
	 * WARNING: Always call this directly. Always signal with direct connection.
	 * abort() must execute in the GUI thread!
	 */
	void abort();
	void deleteLater();

protected:
	virtual void run();

signals:
	/** Label icons computed successfully. */
	void labelIconsComputed(QVector<QImage> icons);

private:
	IconTaskCtxPtr ctxp;
	volatile bool abortFlag;
    tbb::task_group_context tbbTaskGroupContext;
};

#endif // ICONTASK_H
