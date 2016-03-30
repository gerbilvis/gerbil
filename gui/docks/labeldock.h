#ifndef LABELDOCK_H
#define LABELDOCK_H

#include <QDockWidget>
#include <opencv2/core/core.hpp>

#include <QModelIndex>
#include <QListView>

#include <model/representation.h>
#include <shared_data.h>

namespace Ui {
class LabelDock;
class LabelView;
}

class QStandardItemModel;
class QStandardItem;
class QItemSelection;

class AutohideView;
class AutohideWidget;
class QGraphicsScene;
class QGraphicsProxyWidget;

// TODO: add label for icon size: XXXX px

class LabelDock : public QDockWidget
{
	Q_OBJECT

public:

	explicit LabelDock(QWidget *parent = 0);
	~LabelDock();

	/** Set the size of the multi_img. */
	void setImageSize(cv::Size imgSize);
	void restoreState();
	
public slots:

	void setLabeling(const cv::Mat1s &labels,
	                 const QVector<QColor>& colors,
	                 bool colorsChanged);

	void processPartialLabelUpdate(const cv::Mat1s &, const cv::Mat1b &);

	/** Merge/delete currently selected labels depending on sender. */
	void mergeOrDeleteSelected();

	void processMaskIconsComputed(QVector<QImage> icons);

	void toggleLabelSelection(int label, bool innerSource = false);

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
	void toggleLabelHighlightRequested(short label);

	/** Request a vector of mask icons representing the label masks. */
	void labelMaskIconsRequested(QSize size = QSize());

	/** The user changed the mask icon size using the slider. */
	void labelMaskIconSizeChanged(const QSize& size);

	/** applyROI toggled. */
	void applyROIChanged(bool applyROI);

	/** Request open load labels from file dialog. */
	void requestLoadLabeling();
	/** Request open save labels to file dialog. */
	void requestSaveLabeling();

protected:

	void resizeEvent(QResizeEvent * event);
	void showEvent(QShowEvent *event);

private slots:

	void processSelectionChanged(const QItemSelection & selected,
	                             const QItemSelection & deselected);
	void processLabelItemSelectionChanged(QModelIndex midx);

	void processApplyROIToggled(bool checked);
	
	// from ImageModel
	void processRoiRectChanged(cv::Rect newRoi);

	void updateLabelIcons();

	void updateSliderToolTip();

	/** Adjust view contents size. */

	void resizeSceneContents();

	void deselectSelectedLabels();

	void saveState();

private:

	enum { LabelIndexRole = Qt::UserRole };

	void init();

	// UI with autohide widgets.
	// The view and scene for this widget.
	AutohideView   *ahview;
	QGraphicsScene *ahscene;
	// The Qt Designer generated UI of the dockwidget. The top and bottom autohide
	// widgets are contained here-in but are reparented in init().
	Ui::LabelDock  *ui;
	// Widget in ahscene for main Ui (i.e. labelView).
	QGraphicsProxyWidget *mainUiWidget;
	AutohideWidget *ahwidgetTop;
	AutohideWidget *ahwidgetBottom;

	// The Qt model for the label view.
	// Note: This is _not_ a gerbil model.
	QStandardItemModel *labelModel;

	bool hovering;
	short hoverLabel;

	// Label mask icons indexed by labelid
	QVector<QImage> icons;

	// Label colors indexed by labelid
	QVector<QColor> colors;

	// The size of the multi_img
	cv::Size imgSize;

	// The current ROI.
	cv::Rect roi;
};

#endif // LABELDOCK_H
