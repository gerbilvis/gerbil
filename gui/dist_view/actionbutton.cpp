#include "actionbutton.h"

void ActionButton::processCheck()
{
	if (!ownSignal) {
		setChecked(!isChecked());
	}
	ownSignal = false;
}

void ActionButton::processAction()
{
	ownSignal = true;
	emit passAction();
}

void ActionButton::setAction(QAction *action)
{
	if (actionOwner != nullptr) {
		disconnect(this, SIGNAL(clicked()), this, SLOT (processAction()));
		disconnect(this, SIGNAL(passAction()), actionOwner, SLOT(trigger()));
		disconnect(actionOwner, SIGNAL(triggered(bool)), this, SLOT(processCheck()));
	}
	actionOwner = action;
	connect(this, SIGNAL(clicked()), this, SLOT (processAction()));
	connect(this, SIGNAL(passAction()), actionOwner, SLOT(trigger()));
	connect(actionOwner, SIGNAL(triggered(bool)), this, SLOT(processCheck()));
}
