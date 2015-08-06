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

signals:
	void passAction();

public slots:

	void processCheck();
	void processAction();

private:
	QAction *actionOwner;
	bool ownSignal = false;

};

#endif // ACTIONBUTTON_H
