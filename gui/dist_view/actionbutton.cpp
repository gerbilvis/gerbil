#include "actionbutton.h"


void ActionButton::setAction(QAction *action)
{
	if (actionOwner != nullptr) {
		disconnect(actionOwner, SIGNAL(toggled(bool)), this, SLOT(updateButtonStatusFromAction()));
		disconnect(this, SIGNAL(clicked()), actionOwner, SLOT(trigger()));
	}

	actionOwner = action;
	updateButtonStatusFromAction();
	connect(action, SIGNAL(toggled(bool)), this, SLOT(updateButtonStatusFromAction()));
	connect(this, SIGNAL(clicked()), actionOwner, SLOT(trigger()));
}

void ActionButton::updateButtonStatusFromAction()
{
	setCheckable(actionOwner->isCheckable());
	setChecked(actionOwner->isChecked());
}
