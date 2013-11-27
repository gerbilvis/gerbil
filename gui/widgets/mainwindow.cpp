/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/mainwindow.h"
#include "controller/controller.h"
#include "controller/distviewcontroller.h"
#include "iogui.h"

/*#include "tasks/rgbtbb.h"
#include "tasks/normrangecuda.h"
#include "tasks/normrangetbb.h"
#include <background_task/background_task_queue.h>
*/

#include "docks/illumdock.h"
#include "docks/clusteringdock.h"

#include <labeling.h>
#include <qtopencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <iostream>
#include <QShortcut>
#include <QFileInfo>

MainWindow::MainWindow(bool limitedMode)
	: limitedMode(limitedMode),
	  contextMenu(NULL)
{
	// create all objects
	setupUi(this);
}

void MainWindow::initUI(std::string filename)
{
	/* set title */
	QFileInfo fi(QString::fromStdString(filename));
	setWindowTitle(QString("Gerbil - %1").arg(fi.completeBaseName()));
}

void MainWindow::initSignals(Controller *chief, DistViewController *chief2)
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
			singleButton, SLOT(setDisabled(bool)));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			chief, SIGNAL(toggleIgnoreLabels(bool)));
	connect(singleButton, SIGNAL(toggled(bool)),
			chief, SIGNAL(toggleSingleLabel(bool)));

	// label manipulation from current dist_view
	connect(addButton, SIGNAL(clicked()),
			chief2, SLOT(addHighlightToLabel()));
	connect(remButton, SIGNAL(clicked()),
			chief2, SLOT(remHighlightFromLabel()));

	connect(markButton, SIGNAL(toggled(bool)),
			chief2, SIGNAL(toggleLabeled(bool)));
	connect(nonmarkButton, SIGNAL(toggled(bool)),
			chief2, SIGNAL(toggleUnlabeled(bool)));

//	connect(chief2, SIGNAL(normTargetChanged(bool)),
//			this, SLOT(normTargetChanged(bool)));

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

void MainWindow::setGUIEnabled(bool enable, TaskType tt)
{
	ignoreButton->setEnabled(enable || tt == TT_TOGGLE_LABELS);
	addButton->setEnabled(enable);
	remButton->setEnabled(enable);
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
	cv::Mat output = vole::QImage2Mat(shot.toImage());

	IOGui io("Screenshot File", "screenshot", this);
	io.writeFile(QString(), output);
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
