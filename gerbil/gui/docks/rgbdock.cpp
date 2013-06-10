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

void RgbDock::updatePixmap(coloring type, QPixmap p)
{
//	GGDBG_CALL();
	view->setEnabled(true);
	view->setPixmap(p);
	view->update();
	rgbValid=true;

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
	view->setEnabled(false);

	connect(this, SIGNAL(visibilityChanged(bool)),
			this, SLOT(processVisibilityChanged(bool)));
}

void RgbDock::processVisibilityChanged(bool visible)
{
	dockVisible = visible;
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	if(dockVisible && !rgbValid) {
		//GGDBGM("requesting rgb"<<endl);
		view->setEnabled(false);
		emit rgbRequested(CMF);
	}
}

void RgbDock::processImageUpdate(representation::t type, SharedMultiImgPtr)
{
	//GGDBGM(format("visible=%1%  rgbValid=%2%")%dockVisible %rgbValid <<endl);
	if(representation::IMG == type) {
		rgbValid = false;
		view->setEnabled(false);
		if(dockVisible) {
			//GGDBGM("requesting rgb"<<endl);
			emit rgbRequested(CMF);
		}
	}
}
