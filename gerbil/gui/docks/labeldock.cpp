

#include "labeldock.h"
#include "ui_labeldock.h"

#include <QStandardItemModel>

#include <iostream>
#include "../gerbil_gui_debug.h"

LabelDock::LabelDock(QWidget *parent) :
    QDockWidget(parent),
	ui(new Ui::LabelDock),
	labelModel(new QStandardItemModel)
{
	ui->setupUi(this);
	init();
}

void LabelDock::init()
{
	ui->labelView->setModel(labelModel);
	ui->labelView->setFlow(QListView::LeftToRight);
	ui->labelView->setMovement(QListView::Static);
	ui->labelView->setResizeMode(QListView::Adjust);
	ui->labelView->setViewMode(QListView::IconMode);
	ui->labelView->setSelectionMode(QAbstractItemView::ExtendedSelection);
	ui->labelView->setSelectionRectVisible(true);
	// prevent editing of items
	ui->labelView->setEditTriggers(QAbstractItemView::NoEditTriggers);

	connect(ui->labelView->selectionModel(),
			SIGNAL(selectionChanged(QItemSelection,QItemSelection)),
			this,
			SLOT(processSelectionChanged(QItemSelection,QItemSelection)));


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

	GGDBGM("received new labeling" <<endl);

	// selection will be gone -> disable merge button
	ui->mergeBtn->setDisabled(true);
	labelModel->clear();

	// FIXME not handling colorsChanged here!
	// If only the colors changed, we could keep the current selection.

	GGDBGM(colors.size() <<" colors"<<endl);
	if(colors.size() < 1) {
		// only background, no "real" labels
		return;
	}

	for(int i=1; i<colors.size(); i++) {
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

	foreach(QModelIndex idx, modelIdxs) {
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
