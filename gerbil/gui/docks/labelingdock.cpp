#include "labelingdock.h"
#include "ui_labelingdock.h"

LabelingDock::LabelingDock(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::LabelingDock)
{
	ui->setupUi(this);
}

LabelingDock::~LabelingDock()
{
	delete ui;
}
