#include <QVBoxLayout>

#include <iostream>
#include "gerbil_gui_debug.h"
#include "../scaledview.h"
#include "rgbdock.h"

RgbDock::RgbDock(QWidget *parent) :
    QDockWidget(parent)
{
	initUi();
}

void RgbDock::updatePixmap(QPixmap p)
{
	GGDBG_CALL();
	view->setPixmap(p);
	view->update();
}

void RgbDock::initUi()
{
	setWindowTitle("RGB View");
	QWidget *child = new QWidget(this);
	QVBoxLayout *layout = new QVBoxLayout();
	view = new ScaledView();
	layout->addWidget(view);
	child->setLayout(layout);
	child->setSizePolicy(
				QSizePolicy::Expanding,
				QSizePolicy::Expanding);
	setWidget(child);
	child->setVisible(true);
}
