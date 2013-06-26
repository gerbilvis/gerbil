#include "banddock.h"
#include "ui_banddock.h"

BandDock::BandDock(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::BandDock)
{
	ui->setupUi(this);
}

BandDock::~BandDock()
{
	delete ui;
}
