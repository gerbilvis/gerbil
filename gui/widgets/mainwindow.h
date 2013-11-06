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
#include "docks/roidock.h"
#include <shared_data.h>
#include <multi_img.h>
#include <labeling.h>
// TODO: should belong to a controller
#include <model/illuminationmodel.h>
#include <progress_observer.h>
#include "commandrunner.h"
#ifdef WITH_SEG_MEANSHIFT
#include <meanshift_shell.h>
#endif
#ifdef WITH_SEG_MEDIANSHIFT
#include <medianshift_shell.h>
#endif
#ifdef WITH_SEG_PROBSHIFT
#include <probshift_shell.h>
#endif
#include <graphseg.h>

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QMenu>
#include <opencv2/core/core.hpp>

class Controller;
class DistViewController;

class MainWindow : public QMainWindow, private Ui::MainWindow {
    Q_OBJECT
public:
	MainWindow(bool limitedMode = false);
	void initUI(std::string filename);
	void initSignals(Controller *chief, DistViewController *chief2);

	// add distribution view widget to the appropr. container
	void addDistView(QWidget *frame);

	void setGUIEnabled(bool enable, TaskType tt = TT_NONE);

public slots:
	void openContextMenu();

	void screenshot();

protected:
	void changeEvent(QEvent *e);

private:
	QMenu *contextMenu;

	// only limited full_image available
	bool limitedMode;
};

#endif // MAINWINDOW_H
