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

	/* TODO: old(from johannes, not sure what this is to mean):
	 * We could think about data-sharing between image model
	 * and falsecolor model for the CMF part.
	 */
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

void RgbDock::processVisibilityChanged(bool visible)
{
	if(visible && !rgbValid) {
		emit rgbRequested();
	}
}

void RgbDock::processImageUpdate(representation::t type, SharedMultiImgPtr)
{
	if(representation::IMG == type) {
		rgbValid = false;
		if(this->isVisible()) {
			emit rgbRequested();
		}
	}
}
