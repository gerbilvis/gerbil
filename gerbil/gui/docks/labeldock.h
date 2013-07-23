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
	/** Merge currently selected labels. */
	void mergeSelected();

signals:
	/** The user has selected labels and want's them to be merged. */
	void mergeLabelsRequested(const QVector<int>& labels);

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

private slots:
	void processSelectionChanged(const QItemSelection & selected,
							const QItemSelection & deselected);
	void processLabelItemEntered(QModelIndex midx);
	void processLabelItemLeft();

private:

	enum { LabelIndexRole = Qt::UserRole };

	void init();
	void addLabel(int idx, const QColor& color);


	Ui::LabelDock *ui;

	// The Qt model for the label view.
	// (Note: This is _not_ a gerbil model)
	QStandardItemModel *labelModel;

	bool hovering;
	short hoverLabel;

//	cv::Mat1s labels;
//	QVector<QColor> colors;
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
