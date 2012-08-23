#include "resampling_detection.h"
#include "resampling_detection_gui.h"
#include <QAction>

CommandWrapper* resampling_detection_gui::spawn(VoleGui *parent) {
	// load an image or fail
	std::pair<QString, QImage*> img = openImage();
	if (img.second)
		return new resampling_detection_gui(parent, img);
	else
		return NULL;
}

resampling_detection_gui::resampling_detection_gui(VoleGui *parent, std::pair<QString, QImage*> img)
	: CommandWrapper(parent, img) {

	// read in config files
	findConfFiles("/home/riess/vole/trunk/resampling/configs/");

	// connect execution thread to main thread
	connect(&ex, SIGNAL(imgStart()), this, SLOT(workStarted()));
	connect(&ex, SIGNAL(imgDone(QImage*)), this, SLOT(workStopped(QImage*)));

	// ready for action
	createGUI(parent);

	numResTabs = 0;
}

resampling_detection_gui::~resampling_detection_gui() { delete Image; }

void resampling_detection_gui::createGUI(QWidget * /*parent*/) {
	// connect tab click with update
	connect(container, SIGNAL(currentChanged(int)), this, SLOT(updateTab()));

	// area selection tab
	container->addTab(createSelectionTab(), "Resampling Detection");
}

void resampling_detection_gui::findConfFiles(const QString& path) {
	QDir dir(path);
	dir.setFilter(QDir::Files | QDir::Hidden);

	QFileInfoList list = dir.entryInfoList();
	for (int i = 0; i < list.size(); ++i) {
		QFileInfo fileInfo = list.at(i);
		confFiles.append(fileInfo.filePath());
		confNames.append(fileInfo.fileName());
	}
}

QWidget* resampling_detection_gui::createSelectionTab() {
	// create a scroll area for the input image
	scaImg = new ScrollArea();
	scaImg->setImage(Image);

	// create fancy buttons
	btnZoomIn = new QPushButton("+");
	btnZoomOut = new QPushButton("-");
	btnExecute = new QPushButton("Execute");
	btnClear = new QPushButton("Clear");
	btnClear->setEnabled(false);
	
	// connect the buttons
	connect(btnZoomIn, SIGNAL(clicked()), scaImg, SLOT(zoomIn()));
	connect(btnZoomOut, SIGNAL(clicked()), scaImg, SLOT(zoomOut()));
	connect(btnExecute, SIGNAL(clicked()), this, SLOT(startProcess()));
	connect(btnClear, SIGNAL(clicked()), this, SLOT(clearPicture()));

	// config stuff
	QLabel *label = new QLabel();
	label->setText("Parameters:");

	selConfig = new QComboBox();
	selConfig->addItems(confNames);
	
	// create selection areas
	// A1
	scaImg->addRectangle(QRect(20, 20, 50, 50), QString("A"), QString("lblRect1"));
	// A2
	scaImg->addRectangle(QRect(20, 100, 50, 50), QString("B"), QString("lblRect2"));


	// finally put everything together
	selTab = new QWidget();

    QVBoxLayout *verticalLayout = new QVBoxLayout();	
	verticalLayout->addWidget(btnZoomIn);
	verticalLayout->addWidget(btnZoomOut);
    verticalLayout->addStretch();
	verticalLayout->addWidget(label);
	verticalLayout->addWidget(selConfig);
	verticalLayout->addWidget(btnExecute);
	verticalLayout->addWidget(btnClear);
    verticalLayout->addStretch();

	QHBoxLayout *horizontalLayout = new QHBoxLayout(selTab);
	horizontalLayout->addLayout(verticalLayout);
	horizontalLayout->addWidget(scaImg);

	return selTab;
}

QWidget* resampling_detection_gui::createResultTab(QImage *resImg) {
	// create a new tab to show the results
	ScrollArea *scaNeu;
	QWidget    *tab = new QWidget();

	// create a scroll area for the input image
	scaNeu = new ScrollArea(tab);
	scaNeu->setGeometry(10, 20, scaImg->width(), scaImg->height());
	scaNeu->setObjectName("scaRes");

	// close button, to close the tab
	QPushButton *btnClose = new QPushButton("Close", tab);
	btnClose->setObjectName("btnClose");
	connect(btnClose, SIGNAL(clicked()), this, SLOT(closeTab()));

/*
	QPushButton *btnZoomIn = new QPushButton( "+", tab );
	btnZoomIn->move( ui->btnZoomIn->x(), ui->btnZoomIn->y() );
	btnZoomIn->resize( ui->btnZoomIn->width(), ui->btnZoomIn->height() );
	connect( btnZoomIn, SIGNAL( clicked() ), scaNeu, SLOT( zoomIn() ) );

	QPushButton *btnZoomOut = new QPushButton( "-", tab );
	btnZoomOut->move( ui->btnZoomOut->x(), ui->btnZoomOut->y() );
	btnZoomOut->resize( ui->btnZoomOut->width(), ui->btnZoomOut->height() );
	connect( btnZoomOut, SIGNAL( clicked() ), scaNeu, SLOT( zoomOut() ) );
 */

	// show the result image
	scaNeu->setImage(resImg);

	// save reference to result tab
	resTabs[numResTabs++] = tab;

	return tab;
}

void resampling_detection_gui::clearPicture() {
	scaImg->setImage(Image);
	btnClear->setEnabled(false);
}

void resampling_detection_gui::workStarted() {
	statusBar->showMessage(tr("Image processing started ..."));
}

void resampling_detection_gui::workStopped(QImage *img) {
	// set the state
	statusBar->showMessage(tr("Image processing done."));

	scaImg->setImage(img);

	btnExecute->setEnabled(true);
	btnClear->setEnabled(true);

	// create a new tab for result
	//container->addTab(createResultTab(&img), "blackjack");
}

void resampling_detection_gui::closeEvent(QCloseEvent *ev) {
	qDebug() << "resampling_detection_gui::close event ";

	if (ex.isRunning()) {
		//ex.stopExecution();
		// warte auf thread
		//ex.wait();
	}

	// close application
	ev->accept();
}

void resampling_detection_gui::startProcess() {
	std::cout << "action" << std::endl;
	ResamplingDetection *worker = new ResamplingDetection();

	if (confFiles.size() > 0) {
		QString path = confFiles.at(selConfig->currentIndex());
		bool success = parseOptionsDescription(worker->getOptions(), path.toLocal8Bit());

		if (!success) {
			delete worker;
			return;
		}
	} else {
		std::cerr << "ResamplingDetectionGui::startProcess: no config files known to me :/" << std::endl;
	}
	
	btnExecute->setEnabled(false);

	// now add what the user has set: image, ROIs
	worker->global_config.inputfile = imgName.toStdString();
	
	QList<Rectangle*> &rectList = scaImg->getRectList();
	std::vector<unsigned int>& roi = worker->config.b.rio;
	for (int i = 0; i < rectList.size(); ++i) {
		QRect r = scaImg->getRect(i);
		roi.push_back(r.left());
		roi.push_back(r.top());
		roi.push_back(r.right());
		roi.push_back(r.bottom());
	}
	
	ex.setWorker(worker);
	ex.start();
}

