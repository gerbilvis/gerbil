#ifndef LABELDOCK_H
#define LABELDOCK_H

#include <QDockWidget>
#include <opencv2/core/core.hpp>

namespace Ui {
class LabelDock;
}

class QStandardItemModel;
class QStandardItem;
class QItemSelection;


class LabelDock : public QDockWidget
{
	Q_OBJECT
	
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

private slots:
	void processSelectionChanged(const QItemSelection & selected,
							const QItemSelection & deselected);

private:

	enum { LabelIndexRole = Qt::UserRole };

	void init();
	void addLabel(int idx, const QColor& color);


	Ui::LabelDock *ui;

	// The Qt model for the label view.
	// (Note: This is _not_ a gerbil model)
	QStandardItemModel *labelModel;

//	cv::Mat1s labels;
//	QVector<QColor> colors;
};

#endif // LABELDOCK_H
