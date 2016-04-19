#include "labeldock.h"
#include "ui_labeldock.h"

#include <QStandardItemModel>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsWidget>
#include <QGraphicsLayout>
#include <QSettings>
#include <QDebug>

#include "../widgets/autohideview.h"
#include "../widgets/autohidewidget.h"

#include <gerbil_ostream_ops.h>

#include <iostream>
#include <algorithm>

//#define GGDBG_MODULE
#include "../gerbil_gui_debug.h"


LabelDock::LabelDock(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::LabelDock),
    labelModel(new QStandardItemModel),
    hovering(false),
    hoverLabel(-1)
{
	setObjectName("LabelDock");
	connect(QApplication::instance(), SIGNAL(lastWindowClosed()),
	        this, SLOT(saveState()));
	init();
	restoreState();
}

void LabelDock::init()
{
	ahscene = new QGraphicsScene();

	QWidget *mainUiWidgetTmp = new QWidget();
	ui->setupUi(mainUiWidgetTmp);
	mainUiWidget = ahscene->addWidget(mainUiWidgetTmp);
	// TODO: origin of this problem not understood
	mainUiWidget->setTransform(QTransform::fromTranslate(-9, -9));

	ui->labelView->setModel(labelModel);

	connect(ui->labelView->selectionModel(),
	        SIGNAL(selectionChanged(QItemSelection,QItemSelection)),
	        this,
	        SLOT(processSelectionChanged(QItemSelection,QItemSelection)));

	connect(ui->mergeBtn, SIGNAL(clicked()),
	        this, SLOT(mergeOrDeleteSelected()));
	connect(ui->delBtn, SIGNAL(clicked()),
	        this, SLOT(mergeOrDeleteSelected()));
	connect(ui->consolidateBtn, SIGNAL(clicked()),
	        this, SLOT(deselectSelectedLabels()));
	connect(ui->consolidateBtn, SIGNAL(clicked()),
	        this, SIGNAL(consolidateLabelsRequested()));

	connect(ui->sizeSlider, SIGNAL(valueChanged(int)),
	        this, SLOT(updateLabelIcons()));
	// Icon size is hard-coded to the range [4, 1024] in IconTask.
	ui->sizeSlider->setMinimum(std::max(4, ui->sizeSlider->minimum()));
	ui->sizeSlider->setMaximum(std::min(1024, ui->sizeSlider->maximum()));

	connect(ui->applyROI, SIGNAL(toggled(bool)),
	        this, SLOT(processApplyROIToggled(bool)));

	connect(ui->loadLabelingButton, SIGNAL(clicked()),
	        this, SIGNAL(requestLoadLabeling()));
	connect(ui->saveLabelingButton, SIGNAL(clicked()),
	        this, SIGNAL(requestSaveLabeling()));

	connect(this, SIGNAL(visibilityChanged(bool)),
	        this, SLOT(resizeSceneContents()));

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

void LabelDock::updateLabelIcons()
{
	//GGDBGM("value " << ui->sizeSlider->value() << endl);
	int size = ui->sizeSlider->value();
	size += (size % 2); // avoid uneven sizes due to oscillating interpolation

	// aspect ratio
	float r = 1.0;
	if (imgSize != cv::Size()) {
		r = float(imgSize.height) / float(imgSize.width);
		GGDBGM("image aspect ratio " << r << endl);
	}
	if (ui->applyROI->isChecked() && roi != cv::Rect() && roi.width > 0) {
		r = float(roi.height) / roi.width;
		GGDBGM("ROI aspect ratio " << r << endl);
	}

	QSize iconSize(size, size);
	if (r <= 1.0f) {
		iconSize.setHeight(int(size * r + 0.5));
	} else {
		iconSize.setWidth(int(size / r + 0.5));
	}

	GGDBGM("ratio " << r << ", icon size " << iconSize << endl);

	updateSliderToolTip();
	emit labelMaskIconsRequested(iconSize);
}

LabelDock::~LabelDock()
{
	delete ui;
}

void LabelDock::setImageSize(cv::Size imgSize)
{
	this->imgSize = imgSize;
}

void LabelDock::setLabeling(const cv::Mat1s & labels,
                            const QVector<QColor> &colors,
                            bool colorsChanged)
{
	//GGDBGM("colors.size()=" << colors.size()
	//	   << "  colorsChanged=" << colorsChanged << endl;)
	if (colors.size() < 1) {
		// only background, no "real" labels
		return;
	}

	// did the colors change?
	// FIXME: It is probably uneccessary to compare the color vectors.
	// Test after 1.0 release.
	if (colorsChanged || (this->colors != colors) ) {
		this->colors = colors;
	}

	if (colors.size() != labelModel->rowCount()) {
		// label count changed, slection invalid
		ui->mergeBtn->setDisabled(true);
		ui->delBtn->setDisabled(true);
	}

	deselectSelectedLabels();

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
	for (auto idx : modelIdxs) {
		int id = idx.data(LabelIndexRole).value<int>();
		selectedLabels.push_back(id);
	}

	// Tell the LabelingModel:
	if (sender() == ui->delBtn) {
		emit deleteLabelsRequested(selectedLabels);
	} else
	{
		emit mergeLabelsRequested(selectedLabels);
	}

	emit labelMaskIconsRequested();
}

void LabelDock::deselectSelectedLabels()
{
	for(auto &idx : ui->labelView->selectionModel()->selectedIndexes())
	{
		int id = idx.data(LabelIndexRole).value<int>();
		toggleLabelSelection(id, true);
	}
}

void LabelDock::processMaskIconsComputed(QVector<QImage> icons)
{
	//GGDBG_CALL();
	this->icons = icons;

	bool rebuild = false;
	if (icons.size() != labelModel->rowCount()) {
		// the label count has changed
		//GGDBGM("label count has changed" << endl);
		labelModel->clear();
		rebuild = true;
	}

	if (icons.size()>0) {
		ui->labelView->setIconSize(icons[0].size());
	}

	// no tree model -> just iterate flat over all items
	for (int i = 0; i < icons.size(); i++) {

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
		QModelIndex idx = labelModel->index(i, 0);

		labelModel->setData(idx, icon, Qt::DecorationRole);
	}
}

void LabelDock::resizeEvent(QResizeEvent *event)
{
	resizeSceneContents();
	QDockWidget::resizeEvent(event);
}

void LabelDock::showEvent(QShowEvent *event)
{
	QDockWidget::showEvent(event);
	// Without this, the labelView (view contents) is not properly sized.
	resizeSceneContents();
}

void LabelDock::processSelectionChanged(const QItemSelection &selected,
                                        const QItemSelection &deselected)
{
	int nSelected = ui->labelView->selectionModel()->selectedIndexes().size();

	// more than one label selected
	ui->mergeBtn->setEnabled(nSelected > 1);
	// any label selected
	ui->delBtn->setEnabled(nSelected > 0);

	for (auto &item : selected.indexes()) {
		processLabelItemSelectionChanged(item);
	}

	for (auto &item : deselected.indexes()) {
		processLabelItemSelectionChanged(item);
	}
}

void LabelDock::processLabelItemSelectionChanged(QModelIndex midx)
{
	short label = midx.data(LabelIndexRole).value<int>();
	//GGDBGM("hovering over " << label << endl);
	hovering = true;
	hoverLabel = label;
	emit toggleLabelHighlightRequested(label);
}

void LabelDock::toggleLabelSelection(int label, bool innerSource)
{
	hovering = true;
	hoverLabel = label;

	if(!innerSource) this->blockSignals(true); //to prevent feedback

	QModelIndex index = ui->labelView->model()->index(label, 0);
	if (index.isValid() ) {
		if (ui->labelView->selectionModel()->isSelected(index)) {
			ui->labelView->selectionModel()->select(index, QItemSelectionModel::Deselect);
		} else {
			ui->labelView->selectionModel()->select(index, QItemSelectionModel::Select);
		}
	}

	if(!innerSource) this->blockSignals(false);
}

void LabelDock::processApplyROIToggled(bool checked)
{
	const bool applyROI = checked;
	emit applyROIChanged(applyROI);
	updateLabelIcons();
}

void LabelDock::processRoiRectChanged(cv::Rect newRoi)
{
	GGDBG_CALL();
	roi = newRoi;
	updateLabelIcons();
}

void LabelDock::updateSliderToolTip()
{
	QString t = QString("Icon Size (%1)").arg(ui->sizeSlider->value());
	ui->sizeSlider->setToolTip(t);
}

void LabelDock::resizeSceneContents()
{
	if (!(mainUiWidget && ahview)) {
		return;
	}

	// resize labelView
	/* not sure why the +1 is needed */
	QRect geom = ahview->geometry();
	geom.adjust(0, 0, +1, 0);

	/* if we are "big enough", let ahview scroll-in the widgets */
	if (geom.height() > 550) {
		geom.adjust(0, ahwidgetTop->height(), 0, 0);
	} else {
		geom.adjust(0, AutohideWidget::OutOffset, 0, 0);
	}

	if (geom.height() > 300) {
		geom.adjust(0, 0, 0, -ahwidgetBottom->height());
	} else {
		geom.adjust(0, 0, 0, -AutohideWidget::OutOffset);
	}

	mainUiWidget->setGeometry(geom);
	ahview->fitContentRect(geom.adjusted(0, 0, 0, -AutohideWidget::OutOffset));
}

void LabelDock::saveState()
{
	QSettings settings;
	settings.setValue("Labeling/iconsSize", ui->sizeSlider->value());
	settings.setValue("Labeling/applyROI", ui->applyROI->isChecked());
}

void LabelDock::restoreState()
{
	QSettings settings;
	auto roiChecked = settings.value("Labeling/applyROI", true);
	auto size = settings.value("Labeling/iconsSize", 64);

	ui->sizeSlider->setValue(size.toInt());
	ui->applyROI->setChecked(roiChecked.toBool());
}
