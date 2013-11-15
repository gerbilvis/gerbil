#ifndef AHCOMBOBOX_H
#define AHCOMBOBOX_H

#include <QComboBox>
#include <QMenu>
#include <QVector>

class AutohideView;

class AHComboBox : public QComboBox
{
	Q_OBJECT
public:
	explicit AHComboBox(QWidget *parent = 0);

	void setAHView(AutohideView *v) { view = v; }

	void showPopup();
	void hidePopup();

signals:

public slots:

protected:
	void populateMenu();

	QMenu menu;
	QVector<QAction*> actions;
	AutohideView *view;
};

#endif // AHCOMBOBOX_H
