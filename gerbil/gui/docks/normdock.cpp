#include "normdock.h"
#include "ui_normdock.h"

NormDock::NormDock(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::NormDock)
{
	ui->setupUi(this);
}

NormDock::~NormDock()
{
	delete ui;
}
