/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ui_mainwindow.h"


class MainWindow : public QMainWindow, private Ui::MainWindow {
    Q_OBJECT
public:
	MainWindow();
	void initUI(const QString &filename);
	void initSignals(QObject *ctrl, QObject *dvctrl);

	// add distribution view widget to the appropr. container
	void addDistView(QWidget *frame);

public slots:
	void openContextMenu();

	void screenshot();

protected:
	void changeEvent(QEvent *e);

private:
	QMenu *contextMenu;
};

#endif // MAINWINDOW_H
