/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/mainwindow.h"

#include <app/gerbilio.h>

#include <docks/illumdock.h>
#include <docks/clusteringdock.h>

#include <qtopencv.h>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <QShortcut>
#include <QFileInfo>
#include <QMenu>
#include <QSettings>

#include <iostream>

MainWindow::MainWindow()
	: contextMenu(NULL)
{
	// create all objects
	setupUi(this);
}

void MainWindow::initUI(const QString &filename)
{
	/* set title */
	QFileInfo fi(filename);
	setWindowTitle(QString("Gerbil - %1").arg(fi.completeBaseName()));

	// restore geometry
	QSettings settings;
	restoreGeometry(settings.value("mainWindow/geometry").toByteArray());
}

void MainWindow::initSignals(QObject *ctrl, QObject *dvctrl)
{
	/* slots & signals: GUI only */
	connect(docksButton, SIGNAL(clicked()),
			this, SLOT(openContextMenu()));

	//	we decided to remove this functionality for now
	//	connect(bandDock, SIGNAL(topLevelChanged(bool)),
	//			this, SLOT(reshapeDock(bool)));

	/* buttons to alter label display dynamics */
	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			ctrl, SIGNAL(toggleIgnoreLabels(bool)));

	// label manipulation from current dist_view
	connect(addButton, SIGNAL(clicked()),
			dvctrl, SLOT(addHighlightToLabel()));
	connect(remButton, SIGNAL(clicked()),
			dvctrl, SLOT(remHighlightFromLabel()));

	connect(markButton, SIGNAL(toggled(bool)),
			dvctrl, SIGNAL(toggleLabeled(bool)));
	connect(nonmarkButton, SIGNAL(toggled(bool)),
			dvctrl, SIGNAL(toggleUnlabeled(bool)));

	//	connect(chief2, SIGNAL(normTargetChanged(bool)),
	//			this, SLOT(normTargetChanged(bool)));

	subscriptionsDebugButton->hide();

	connect(subscriptionsDebugButton, SIGNAL(clicked()),
			ctrl, SLOT(debugSubscriptions()));

	/// global shortcuts
	QShortcut *scr = new QShortcut(Qt::CTRL + Qt::Key_S, this);
	connect(scr, SIGNAL(activated()), this, SLOT(screenshot()));
}

void MainWindow::addDistView(QWidget *frame)
{
	// determine position before spacer (last element in the layout)
	int pos = distviewLayout->count() - 1;
	// add with stretch = 1 so they will stay evenly distributed in space
	distviewLayout->insertWidget(pos, frame, 1);

	/* TODO: the spacer which is now in the .ui would be added like this
	 * previously. If current method fails, reconsider doing this: */
	// vLayout->addStretch(); // align on top when all folded
}

void MainWindow::openContextMenu()
{
	delete contextMenu;
	contextMenu = createPopupMenu();
	contextMenu->exec(QCursor::pos());
}

void MainWindow::screenshot()
{
	// grabWindow reads from the display server, so GL parts are not missing
	QPixmap shot = QPixmap::grabWindow(this->winId());

	// we use OpenCV so the user can expect the same data type support
	cv::Mat output = QImage2Mat(shot.toImage());

	GerbilIO io(this, "Screenshot File", "screenshot");
	io.setFileSuffix(".png");
	io.setFileCategory("Screenshot");
	io.writeImage(output);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
	QSettings settings;
	// store window geometry
	settings.setValue("mainWindow/geometry", saveGeometry());
	// store dock widgets state
	settings.setValue("mainWindow/windowState", saveState());
	QMainWindow::closeEvent(event);
}

void MainWindow::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}
