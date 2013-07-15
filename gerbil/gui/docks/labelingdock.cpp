#include "labelingdock.h"
#include "ui_labelingdock.h"

LabelingDock::LabelingDock(QWidget *parent) :
	QDockWidget(parent)
{
	setupUi(this);
	initUi();
}

LabelingDock::~LabelingDock()
{
}

void LabelingDock::initUi()
{
	connect(loadLabelingButton, SIGNAL(clicked()),
			this, SIGNAL(requestLoadLabeling()));
	connect(saveLabelingButton, SIGNAL(clicked()),
			this, SIGNAL(requestSaveLabeling()));
}
