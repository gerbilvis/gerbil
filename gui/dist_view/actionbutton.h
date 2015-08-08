#ifndef ACTIONBUTTON_H
#define ACTIONBUTTON_H

#include <QPushButton>
#include <QAction>

class ActionButton : public QPushButton
{
	Q_OBJECT

public:
	explicit ActionButton(QWidget* parent=nullptr)
	    :QPushButton(parent), actionOwner(nullptr) {}
	virtual ~ActionButton() {}

	void setAction(QAction* action);

public slots:

	void updateButtonStatusFromAction();

private:
	QAction *actionOwner;

};

#endif // ACTIONBUTTON_H
