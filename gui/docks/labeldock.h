#ifndef LABELDOCK_H
#define LABELDOCK_H

#include <QDockWidget>
#include <opencv2/core/core.hpp>

#include <QModelIndex>

namespace Ui {
class LabelDock;
}

class QStandardItemModel;
class QStandardItem;
class QItemSelection;

class LabelDock : public QDockWidget
{
	Q_OBJECT

	friend class LeaveEventFilter;
public:
	explicit LabelDock(QWidget *parent = 0);
	~LabelDock();
	
public slots:
	void setLabeling(const cv::Mat1s &labels,
					 const QVector<QColor>& colors,
					 bool colorsChanged);

	void processPartialLabelUpdate(const cv::Mat1s &, const cv::Mat1b &);

	/** Merge/delete currently selected labels depending on sender. */
	void mergeOrDeleteSelected();

	void processMaskIconsComputed(const QVector<QImage>& icons);
signals:
	/** The user has selected labels and wants them to be merged. */
	void mergeLabelsRequested(const QVector<int>& labels);

	/** The user has selected labels and wants them to be deleted. */
	void deleteLabelsRequested(const QVector<int>& labels);

	/** The user pressed the clean-up button */
	void consolidateLabelsRequested();

	/** Request to highlight the given label exclusively.
	 *
	 * Only the given label should be highlighted and only if highlight is
	 * true. When the signal is received with highlight set to false,
	 * highlighting should be stopped.
	 *
	 *  @param label The label whichg should be highlighted.
	 *  @param highlight If true highlight the label. Otherwise stop
	 *  highlighting.
	 */
	void highlightLabelRequested(short label, bool highlight);

	/** Request a vector of mask icons representing the label masks. */
	void labelMaskIconsRequested();

	/** The user changed the mask icon size using the slider. */
	void labelMaskIconSizeChanged(const QSize& size);
private slots:
	void processSelectionChanged(const QItemSelection & selected,
							const QItemSelection & deselected);
	void processLabelItemEntered(QModelIndex midx);
	void processLabelItemLeft();

	/** Icon size slider value changed. */
	void processSliderValueChanged(int);
	void updateSliderToolTip();
private:

	enum { LabelIndexRole = Qt::UserRole };

	void init();


	Ui::LabelDock *ui;

	// The Qt model for the label view.
	// (Note: This is _not_ a gerbil model)
	QStandardItemModel *labelModel;

	bool hovering;
	short hoverLabel;

	// Label mask icons indexed by labelid
	QVector<QImage> icons;

//	cv::Mat1s labels;
	QVector<QColor> colors;
};

// utility class to filter leave event
class LeaveEventFilter : public QObject {
	Q_OBJECT
public:
	LeaveEventFilter(LabelDock *parent)
		: QObject(parent)
	{}
protected:
	bool eventFilter(QObject *obj, QEvent *event);
};

#endif // LABELDOCK_H
