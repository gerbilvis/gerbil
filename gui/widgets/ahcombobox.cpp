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
	menu.setActiveAction(actions[currentIndex()]);
	menu.setDefaultAction(actions[currentIndex()]);

	// map to scene coordinates
#ifdef _WIN32 // mapToGlobal() doesn't work correctly (TODO: test qt 5.7)
	auto screenpoint = QCursor::pos();
#else
	auto screenpoint = mapToGlobal(QPoint(0, 0));
#ifdef QT_BROKEN_MAPTOGLOBAL
	screenpoint = view->mapToGlobal(screenpoint);
#endif
#endif

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
	QComboBox::hidePopup(); // reset internal state of the combobox
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
