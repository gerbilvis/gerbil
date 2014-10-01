

#include "labeldock.h"
#include "ui_labeldock.h"

#include <QStandardItemModel>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QGraphicsLayout>

#include "../widgets/autohideview.h"
#include "../widgets/autohidewidget.h"

#include <iostream>

//#define GGDBG_MODULE
#include "../gerbil_gui_debug.h"


LabelDock::LabelDock(QWidget *parent) :
    QDockWidget(parent),
	ui(new Ui::LabelDock),
	labelModel(new QStandardItemModel),
	hovering(false),
	hoverLabel(-1)
{
	init();
}

void LabelDock::init()
{
	ahscene = new QGraphicsScene();

	QWidget *mainUiWidgetTmp = new QWidget();
	ui->setupUi(mainUiWidgetTmp);

	mainUiWidgetTmp->layout()->setContentsMargins(0,0,0,0);
	mainUiWidgetTmp->layout()->setSpacing(0);
	ui->labelView->setFrameStyle(QFrame::NoFrame);
	mainUiWidget = ahscene->addWidget(mainUiWidgetTmp);

	// FIXME: HACK,  where to get the proper values?
	mainUiWidget->translate(-10, 0);

	ui->labelView->setModel(labelModel);

	LeaveEventFilter *leaveFilter = new LeaveEventFilter(this);
	ui->labelView->installEventFilter(leaveFilter);

	ui->labelView->setDragEnabled(false);
	ui->labelView->setDragDropMode(QAbstractItemView::NoDragDrop);

	connect(ui->labelView->selectionModel(),
			SIGNAL(selectionChanged(QItemSelection,QItemSelection)),
			this,
			SLOT(processSelectionChanged(QItemSelection,QItemSelection)));

	connect(ui->labelView, SIGNAL(entered(QModelIndex)),
			this, SLOT(processLabelItemEntered(QModelIndex)));
	connect(ui->labelView, SIGNAL(viewportEntered()),
			this, SLOT(processLabelItemLeft()));

	connect(ui->mergeBtn, SIGNAL(clicked()),
			this, SLOT(mergeOrDeleteSelected()));
	connect(ui->delBtn, SIGNAL(clicked()),
			this, SLOT(mergeOrDeleteSelected()));
	connect(ui->consolidateBtn, SIGNAL(clicked()),
			this, SIGNAL(consolidateLabelsRequested()));

	connect(ui->sizeSlider, SIGNAL(valueChanged(int)),
			this, SLOT(processSliderValueChanged(int)));
	// Icon size is hard-coded to the range [32, 256] in IconTask.
	// Don't change this unless you know what you are doing.
	ui->sizeSlider->setMinimum(32);
	ui->sizeSlider->setMaximum(256);
	updateSliderToolTip();

	connect(ui->applyROI, SIGNAL(toggled(bool)),
			this, SLOT(processApplyROIToggled(bool)));

	this->setWindowTitle("Labels");
	QWidget *contents = new QWidget(this);
	QVBoxLayout *layout = new QVBoxLayout(contents);
	ahview = new AutohideView(contents);
	ahview->init();
	ahview->setBaseSize(QSize(300, 245));
	ahview->setFrameShape(QFrame::NoFrame);
	ahview->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ahview->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	ahview->setScene(ahscene);
	layout->addWidget(ahview);

	// Setup autohide widgets.

	ahwidgetTop = new AutohideWidget();
	// snatch the layout from the placeholder widget
	QLayout *ahwTopLayout = ui->autoHideTop->layout();
	ahwidgetTop->setLayout(ahwTopLayout);
	// remove note label
	ui->ahwidgetTopNoteLabel->setParent(NULL);
	ui->ahwidgetTopNoteLabel->deleteLater();
	// remove placeholder widget
	ui->autoHideTop->setParent(NULL);
	ui->autoHideTop->deleteLater();
	ahview->addWidget(AutohideWidget::TOP, ahwidgetTop);
	// FIXME: has no effect.
	ui->applyROI->setStyleSheet("color: white;");

	ahwidgetBottom = new AutohideWidget();
	// snatch the layout from the placeholder widget
	QLayout *ahwBottomLayout = ui->autoHideBottom->layout();
	ahwidgetBottom->setLayout(ahwBottomLayout);
	// remove note label
	ui->ahwidgetBottomNoteLabel->setParent(NULL);
	ui->ahwidgetBottomNoteLabel->deleteLater();
	// remove placeholder widget
	ui->autoHideBottom->setParent(NULL);
	ui->autoHideBottom->deleteLater();
	ahview->addWidget(AutohideWidget::BOTTOM, ahwidgetBottom);

	// debug cross-hair
	if (false) {
		const QPen pen(Qt::red);
		const int llenh = 5;
		ahscene->addLine(1,-llenh,1,llenh,pen);
		ahscene->addLine(-llenh,1,llenh,1,pen);
	}

	this->setWidget(contents);
}

LabelDock::~LabelDock()
{
	delete ui;
}

void LabelDock::setLabeling(const cv::Mat1s & labels,
							const QVector<QColor> &colors,
							bool colorsChanged)
{
	//GGDBGM("colors.size()=" << colors.size()
	//	   << "  colorsChanged=" << colorsChanged << endl;)
	if(colors.size() < 1) {
		// only background, no "real" labels
		return;
	}

	// did the colors change?
	// FIXME: It is probably uneccessary to compare the color vectors.
	// Test after 1.0 release.
	if(colorsChanged || (this->colors != colors) ) {
		this->colors = colors;
	}

	if(colors.size() != labelModel->rowCount()) {
		// label count changed, slection invalid
		ui->mergeBtn->setDisabled(true);
		ui->delBtn->setDisabled(true);
	}

	//GGDBGM("requesting new label icons" << endl);
	emit labelMaskIconsRequested();
}

void LabelDock::processPartialLabelUpdate(const cv::Mat1s &, const cv::Mat1b &)
{
	emit labelMaskIconsRequested();
}

void LabelDock::mergeOrDeleteSelected()
{
	QItemSelection selection = ui->labelView->selectionModel()->selection();
	QModelIndexList modelIdxs = selection.indexes();

	/* need at least two selected colors to merge, one color to delete */
	if (modelIdxs.size() < (sender() == ui->delBtn ? 1 : 2)) {
		return;
	}

	// Extract label ids from QModelIndex s.
	QVector<int> selectedLabels;
	foreach (QModelIndex idx, modelIdxs) {
		int id = idx.data(LabelIndexRole).value<int>();
		selectedLabels.push_back(id);
	}

	// Tell the LabelingModel:
	if (sender() == ui->delBtn)
		emit deleteLabelsRequested(selectedLabels);
	else
		emit mergeLabelsRequested(selectedLabels);
	emit labelMaskIconsRequested();
}

void LabelDock::processMaskIconsComputed(const QVector<QImage> &icons)
{
	//GGDBG_CALL();
	this->icons = icons;

	bool rebuild = false;
	if(icons.size() != labelModel->rowCount()) {
		// the label count has changed
		//GGDBGM("label count has changed" << endl);
		labelModel->clear();
		rebuild = true;
	}

	if(icons.size()>0) {
		ui->labelView->setIconSize(icons[0].size());
	}

	// no tree model -> just iterate flat over all items
	for(int i=0; i<icons.size(); i++) {

		const QImage &image = icons[i];
		QPixmap pixmap = QPixmap::fromImage(image);
		QIcon icon(pixmap);

		if(rebuild) {
			// labelModel takes ownership of the item.
			QStandardItem *itm =  new QStandardItem();
			//itm->setData(QIcon(),Qt::DecorationRole);
			itm->setData(i, LabelIndexRole);
			labelModel->appendRow(itm);
		}

		// we are using only rows, i == row
		QModelIndex idx = labelModel->index(i,0);

		labelModel->setData(idx, icon ,Qt::DecorationRole);
	}
}

void LabelDock::resizeEvent(QResizeEvent *event)
{
	// FIXME: Missing initial resize for some reason. After manually resizing
	// everything is fine. ???

	//GGDBGM("mainUiWidget "<< mainUiWidget << ", ahview "<< ahview << endl);
	if (mainUiWidget && ahview) {
		// Resize labelView.
		// Not sure where that -1 and +1 comes from --
		// tested on Arch Linux 2014-10-01,
		// qt4 4.8.6-1, xfwm4 4.10.1-1, xorg-server 1.16.1-1
		QRect geom = ahview->geometry();
		const int off = 2 * AutohideWidget::OutOffset - 1;
		geom.adjust(0, 0, +1, -off);
		mainUiWidget->setGeometry(geom);
	}
	QDockWidget::resizeEvent(event);
}

void LabelDock::processSelectionChanged(const QItemSelection &,
		const QItemSelection &)
{
	int nSelected = ui->labelView->selectionModel()->selectedIndexes().size();

	// more than one label selected
	ui->mergeBtn->setEnabled(nSelected > 1);
	// any label selected
	ui->delBtn->setEnabled(nSelected > 0);
}

void LabelDock::processLabelItemEntered(QModelIndex midx)
{
	short label = midx.data(LabelIndexRole).value<int>();
	//GGDBGM("hovering over " << label << endl);
	hovering = true;
	hoverLabel = label;
	emit highlightLabelRequested(label, true);
}

void LabelDock::processLabelItemLeft()
{
	if (hovering) {
		//GGDBGM("hovering left" << endl);
		hovering = false;
		emit highlightLabelRequested(hoverLabel, false);
	}
}

void LabelDock::processApplyROIToggled(bool checked)
{
	const bool applyROI = checked;
	emit applyROIChanged(applyROI);
	emit labelMaskIconsRequested();
}

void LabelDock::processSliderValueChanged(int)
{
	//GGDBGM("value " << ui->sizeSlider->value() << endl);
	int val = ui->sizeSlider->value();
	// Make icon size even. Odd values cause the interpolation to oscillate
    // while changing slider.
	val = val + (val%2);
	QSize iconSize(val,val);
	updateSliderToolTip();
	emit labelMaskIconSizeChanged(iconSize);
	// Label mask icons update is non-blocking, request them for each change.
	emit labelMaskIconsRequested();
}

void LabelDock::updateSliderToolTip()
{
	QString t = QString("Icon Size (%1)").arg(
				ui->sizeSlider->value());
	ui->sizeSlider->setToolTip(t);
}

bool LeaveEventFilter::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::Leave) {
		//GGDBGM("sending leave event" << endl);
		LabelDock *labelDock = static_cast<LabelDock*>(parent());
		labelDock->processLabelItemLeft();
	}
	return false; // continue normal processing of this event
}
