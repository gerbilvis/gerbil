#include "ahcombobox.h"
#include "autohideview.h"

#include <cassert>
#include <iostream>

AHComboBox::AHComboBox(QWidget *parent) :
	QComboBox(parent), view(NULL)
{
}

void AHComboBox::showPopup()
{
	assert(view);
	if (actions.count() != count()) // we are inconsistent
		populateMenu();

	// map to scene coordinates
	QPoint scenepoint = mapToGlobal(QPoint(0, 0));
	// map to screen coordinates
	QPoint screenpoint = view->mapToGlobal(scenepoint);
	menu.setActiveAction(actions[currentIndex()]);
	// would be nice, but has a drawing bug in Qt (text too wide for window)
	// menu.setDefaultAction(actions[currentIndex()]);
	QAction *a = menu.exec(screenpoint, actions[currentIndex()]);
	if (!a)
		return;

	int choice = actions.indexOf(a);
	setCurrentIndex(choice);
	hidePopup();
}

void AHComboBox::hidePopup()
{
	menu.close();
	/* QComboBox somewhat manages to suck-up the mouse release event, so
	   we need to explicitely unlock the scrolling here */
	view->suppressScrolling(false);
}

void AHComboBox::populateMenu()
{
	menu.clear();
	actions.clear();
	for (int i = 0; i < count(); ++i) {
		QAction *tmp = menu.addAction(itemIcon(i), itemText(i));
		// not necessary: we work with the index
		// tmp->setData(itemData(i));
		actions.append(tmp);
	}
}
