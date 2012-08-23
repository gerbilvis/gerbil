#include "copymoveframework.h"
#include "cmf.h"
#include <QAction>

CommandWrapper* cmf::spawn(VoleGui *parent) {
	// load an image or fail
	std::pair<QString, QImage*> img = openImage();
	if (img.second)
		return new cmf(parent, img);
	else
		return NULL;
}

cmf::cmf(VoleGui *parent, std::pair<QString, QImage*> img)
	: CommandWrapper(parent, img) {

	// read in config files
	findConfFiles("/home/jordan/svn/code/vole/trunk/sivichri/configs/");
	findConfFiles("/home/ypnos/computervision/svn/vole/trunk/sivichri/configs/");

	// connect execution thread to main thread
	connect(&ex, SIGNAL(imgStart()), this, SLOT(workStarted()));
	connect(&ex, SIGNAL(imgDone(QImage*)), this, SLOT(workStopped(QImage*)));

	// ready for action
	createGUI(parent);

	numResTabs = 0;
}

cmf::~cmf() { delete Image; }

void cmf::createGUI(QWidget *) {
	// area selection tab
	container->addTab(createSelectionTab(), "Copy/Move Forgery");
}

void cmf::findConfFiles(const QString& path) {
	QDir dir(path);
	dir.setFilter(QDir::Files | QDir::Hidden);

	QFileInfoList list = dir.entryInfoList();
	for (int i = 0; i < list.size(); ++i) {
		QFileInfo fileInfo = list.at(i);
		confFiles.append(fileInfo.filePath());
		confNames.append(fileInfo.fileName());
	}
}

QWidget* cmf::createSelectionTab() {
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
	label->setText("Strategy:");

	selConfig = new QComboBox();
	selConfig->addItems(confNames);
	
	// create selection areas
	// A1
	scaImg->addRectangle(QRect(20, 20, 50, 50), QString("A1"), QString("lblRect1"));
	// A2
//	scaImg->addRect(QRect(20, 100, 50, 50), QString("A2"), QString("lblRect2"));


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

QWidget* cmf::createResultTab(QImage *resImg) {
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

void cmf::clearPicture() {
	scaImg->setImage(Image);
	btnClear->setEnabled(false);
}

void cmf::workStarted() {
	statusBar->showMessage(tr("Image processing started ..."));
}

void cmf::workStopped(QImage *img) {
	// set the state
	statusBar->showMessage(tr("Image processing done."));

	scaImg->setImage(img);

	btnExecute->setEnabled(true);
	btnClear->setEnabled(true);

	// create a new tab for result
	//container->addTab(createResultTab(&img), "blackjack");
}

void cmf::closeEvent(QCloseEvent *ev) {
	qDebug() << "cmf::close event ";

	if (ex.isRunning()) {
		//ex.stopExecution();
		// warte auf thread
		//ex.wait();
	}

	// close application
	ev->accept();
}

void cmf::startProcess() {
	CopyMoveFramework *worker = new CopyMoveFramework();
	QString path = confFiles.at(selConfig->currentIndex());
	
	bool success = parseOptionsDescription(worker->getOptions(), path.toLocal8Bit());

	if (!success) {
		delete worker;
		return;
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

