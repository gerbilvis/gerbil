

#include "labeldock.h"
#include "ui_labeldock.h"

#include <QStandardItemModel>

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


	connect(ui->labelView->selectionModel(),
			SIGNAL(selectionChanged(QItemSelection,QItemSelection)),
			this,
			SLOT(processSelectionChanged(QItemSelection,QItemSelection)));

	connect(ui->labelView, SIGNAL(entered(QModelIndex)),
			this, SLOT(processLabelItemEntered(QModelIndex)));
	connect(ui->labelView, SIGNAL(viewportEntered()),
			this, SLOT(processLabelItemLeft()));


	connect(ui->mergeBtn, SIGNAL(clicked()),
			this, SLOT(mergeSelected()));
}

void LabelDock::addLabel(int idx, const QColor &color)
{
	// The labelModel takes ownership of the item pointer.
	QStandardItem *itm =  new QStandardItem();

	// fixed icon size for now
	QPixmap pixmap(32,32);
	pixmap.fill(color);

	itm->setData(QIcon(pixmap),Qt::DecorationRole);
	itm->setData(idx, LabelIndexRole);

	labelModel->appendRow(itm);
}

LabelDock::~LabelDock()
{
	delete ui;
}

void LabelDock::setLabeling(const cv::Mat1s & labels,
							const QVector<QColor> &colors,
							bool colorsChanged)
{
	// selection will be gone -> disable merge button
	ui->mergeBtn->setDisabled(true);
	labelModel->clear();

	// FIXME not handling colorsChanged here!
	// If only the colors changed, we could keep the current selection.

	if(colors.size() < 1) {
		// only background, no "real" labels
		return;
	}

	// background is added as well so we can merge to background
	addLabel(0, QColor(Qt::black));

	for (int i=1; i<colors.size(); i++) {
		addLabel(i, colors.at(i));
	}
}

void LabelDock::mergeSelected()
{
	QItemSelection selection = ui->labelView->selectionModel()->selection();
	QModelIndexList modelIdxs = selection.indexes();

	// need at least two selected colors
	if (modelIdxs.size() < 2) {
		return;
	}

	// Extract label ids from QModelIndex s.
	QVector<int> selectedLabels;
	QStringList idxsString; // debug

	foreach (QModelIndex idx, modelIdxs) {
		int id = idx.data(LabelIndexRole).value<int>();
		selectedLabels.push_back(id);
		idxsString.append(QString::number(id));
	}

	std::string mergedlabels = idxsString.join(", ").toStdString();
	GGDBGM("merging labels " <<  mergedlabels << endl);

	// Tell the LabelingModel to merge the selected labels.
	emit mergeLabelsRequested(selectedLabels);
}


void LabelDock::processSelectionChanged(const QItemSelection &,
		const QItemSelection &)
{
	int nSelected = ui->labelView->selectionModel()->selectedIndexes().size();
	//GGDBGM("nSelected " << nSelected << endl);
	// more than one label selected
	ui->mergeBtn->setEnabled(nSelected > 1);
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

bool LeaveEventFilter::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == QEvent::Leave) {
		//GGDBGM("sending leave event" << endl);
		LabelDock *labelDock = static_cast<LabelDock*>(parent());
		labelDock->processLabelItemLeft();
	}
	return false; // continue normal processing of this event
}
