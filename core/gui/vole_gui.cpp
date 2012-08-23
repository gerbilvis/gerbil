#include "vole_gui.h"

#include "generated_gui_include_list.h"

#include <QDebug>
#include <QResizeEvent>

VoleGui::VoleGui(QWidget *parent)
: QMainWindow(parent) { // , startMap(this) {
	startMap = new QSignalMapper(this);
	setupUi(this);
	cornerWidget = new QPushButton("Close");
	workspaceContainer->setCornerWidget(cornerWidget);
	cornerWidget->show();
	connect(cornerWidget, SIGNAL(clicked(bool)), this, SLOT(removeTab(bool)));

	// initialize all methods
	#include "gmodules.inc"

	// connect start button menu
	connect(startMap, SIGNAL(mapped(int)), this, SLOT(spawn(int)));
	
	startButton->setMenu(&startMenu);

	// for resizing
	installEventFilter(this);
//	startMap->moveToThread(this->thread());
//	startMap->setParent(this);
//	std::cout << "end of extremely ugly & unclear" << std::endl;
}

VoleGui::~VoleGui() {
	delete cornerWidget;
}

// removes the current tab from the workspace
void VoleGui::removeTab(bool) {
	QWidget *currentTab = workspaceContainer->currentWidget();
//	workspaceContainer->removeTab(workspaceContainer->currentIndex());
	delete currentTab; // valgrind comments strangely about this delete;
}

void VoleGui::spawn(int id) {
	// call spawn function of specified method
	CommandWrapper *res = startActions[id].second(this);
	if (res)
		wrappers.push_back(res);
	std::cout << "pushed" << std::endl;
}

void VoleGui::registerMethod(const QString& name, startFunc spawn) {
	int id = startActions.size();

	startActions.push_back(std::make_pair(new QAction(name, this), spawn));
	QAction *act = startActions[id].first;
	startMenu.addAction(act);

	startMap->setMapping(act, id);
	connect(act, SIGNAL(triggered()), startMap, SLOT(map()));
}

void VoleGui::remove(CommandWrapper* wrapper) {
	wrappers.erase(std::find(wrappers.begin(), wrappers.end(), wrapper));
}

bool VoleGui::eventFilter(QObject* obj, QEvent* ev) {
	if (obj == this) {
		// resize of the main window ?
		if (ev->type() == QEvent::Resize) {
//			QResizeEvent* rev = static_cast<QResizeEvent*>(ev);

			// resize the method widget
//			cmfGui->resizeGUI(rev);
//			ldGui->resizeGUI(rev);
/*
			QResizeEvent* rev = static_cast<QResizeEvent*>( ev );
			// resize wdImage ( tab )
			QSize delta = rev->size() - rev->oldSize();
			ui->wdImage->resize( rev->size().width() - 20, rev->size().height() - 100 );

			// resize scaImg
			scaImg->resize( rev->size().width() - (20 + 70 + 10), rev->size().height() - ( 100 + 60 ) );

			// move the buttons
			ui->btnLoadImg->move( rev->size().width() - 80, ui->btnLoadImg->pos().y());
			ui->btnZoomIn->move( rev->size().width() - 73, ui->btnZoomIn->pos().y() );
			ui->btnZoomOut->move( rev->size().width() - 73, ui->btnZoomOut->pos().y() );

			// LD
			// resize scaImgLD
			scaImgLD->resize( rev->size().width() - (20 + 70 + 10), rev->size().height() - ( 100 + 60 ) );

			// move buttons
			ui->btnLoadImgLD->move( rev->size().width() - 80, ui->btnLoadImgLD->pos().y());
			ui->btnZoomInLD->move( rev->size().width() - 73, ui->btnZoomInLD->pos().y() );
			ui->btnZoomOutLD->move( rev->size().width() - 73, ui->btnZoomOutLD->pos().y() );
			// move slider
			ui->vsldLD->move( rev->size().width() - 66, ui->vsldLD->pos().y() );
 */
			return true;
		}
	}

	return QWidget::eventFilter(obj, ev);
}
