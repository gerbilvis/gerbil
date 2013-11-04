

#include "labeldock.h"
#include "ui_labeldock.h"

#include <QStandardItemModel>
#include <QLabel> // REMOVE

#include <iostream>
#include "../gerbil_gui_debug.h"


LabelDock::LabelDock(QWidget *parent) :
    QDockWidget(parent),
	ui(new Ui::LabelDock),
	labelModel(new QStandardItemModel),
	hovering(false),
	hoverLabel(-1)
{
	ui->setupUi(this);
	init();
}

void LabelDock::init()
{
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
	updateSliderToolTip();
}

LabelDock::~LabelDock()
{
	delete ui;
}

void LabelDock::setLabeling(const cv::Mat1s & labels,
							const QVector<QColor> &colors,
							bool colorsChanged)
{
	if(colors.size() < 1) {
		// only background, no "real" labels
		return;
	}
	bool requestIcons = false;

	// did the colors change?
	if(this->colors != colors ) {
		this->colors = colors;
		// once we use colors to draw icon borders, we have to trigger an update
		// here
		requestIcons = true;
	}

	if(colors.size()  != labelModel->rowCount()) {
		// label count changed, slection invalid
		ui->mergeBtn->setDisabled(true);
		ui->delBtn->setDisabled(true);
		requestIcons = true;
	}

	if(requestIcons) {
		emit labelMaskIconsRequested();
	}
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

void LabelDock::processSliderValueChanged(int)
{
	//GGDBGM("value " << ui->sizeSlider->value() );
	int val = ui->sizeSlider->value();
	// make icon size even, the odd values cause the interpolation  to oscillate
    // while changing slides values.
	val = val + (val%2);
	//GGDBGP(", value  even " << val << endl);
	QSize iconSize(val,val);
	updateSliderToolTip();
	emit labelMaskIconSizeChanged(iconSize);
	// updating label masks is fast, request them for each change.
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
