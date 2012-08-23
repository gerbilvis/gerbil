#include "ld.h"

#include "matrix_operations.h"
#include <global_defines.h>
#include <qtopencv.h>

#include <QTimer>

CommandWrapper* ld::spawn(VoleGui *parent) {
	// load an image or fail
	std::pair<QString, QImage*> img = openImage();
	if (img.second)
		return new ld(parent, img);
	else
		return NULL;
}

ld::ld(VoleGui *parent, std::pair<QString, QImage*> img)
	: CommandWrapper(parent, img) {
	//

	cv::Mat_<cv::Vec3b> input_image = vole::QImage2Mat(img.second);
	ldc = new LightingDirectionCore(input_image);
	// FIXME the threshold 0.2 is hardcoded :(
	double threshold = 0.2;
	// FIXME gray_image zwischenspeichern?
	cv::Mat_<double> gray_image = MatrixOperations::rgb_to_YCbCr(input_image);
	cv::Mat_<double> edge_img_matrix = ldc->get_edge_image(gray_image, threshold);

	kantenImg = Mat2QImage(edge_img_matrix);
	createGUI(parent);
}

ld::~ld() {
	delete ldc;
}

void ld::createGUI(QWidget *) {
	// area selection tab
	container->addTab(createSelectionTab(), "Image part selection");

	// create popup menu
	createPopupMenu();
}

void ld::alphaChanged(int value) {
	blendingImage->alphaChanged(value);
	imagePlane->update();
}


QWidget* ld::createSelectionTab() {
	// create a scroll area for the input image
	QScrollArea *scrollArea = new QScrollArea();
	imagePlane = new ImagePlane();
	imagePlane->setImage(Image);
	blendingImage = new BlendingImage(0.5, kantenImg);
	imagePlane->addPaintEventNotification(blendingImage);

	// create fancy buttons
	btnZoomIn = new QPushButton("+");
	btnZoomOut = new QPushButton("-");
	btnExecute = new QPushButton("Execute");

	// connect the zoom buttons
	connect(btnZoomIn, SIGNAL(clicked()), imagePlane, SLOT(zoomIn()));
	connect(btnZoomOut, SIGNAL(clicked()), imagePlane, SLOT(zoomOut()));
	connect(btnExecute, SIGNAL(clicked()), this, SLOT(startProcess()));

	// create slider
	vsldAlpha = new QSlider();
	vsldAlpha->setObjectName(QString::fromUtf8("vsldAlpha"));
	vsldAlpha->setMinimum(0);
	vsldAlpha->setMaximum(99);
	vsldAlpha->setSingleStep(0);
	vsldAlpha->setValue(49);
	vsldAlpha->setSliderPosition(49);
	vsldAlpha->setTracking(true);
	vsldAlpha->setOrientation(Qt::Vertical);
	vsldAlpha->setTickPosition(QSlider::TicksBothSides);
	vsldAlpha->setTickInterval(3);

	connect(vsldAlpha, SIGNAL(valueChanged(int)), this,
			SLOT(alphaChanged(int)));

	// finally put everything together
	selTab = new QWidget();

    QVBoxLayout *verticalLayout = new QVBoxLayout();	
	verticalLayout->setAlignment(vsldAlpha, Qt::AlignCenter);
	verticalLayout->addWidget(btnZoomIn);
	verticalLayout->addWidget(btnZoomOut);
	verticalLayout->addWidget(vsldAlpha);
	verticalLayout->addWidget(btnExecute);
    verticalLayout->addStretch();

	QHBoxLayout *horizontalLayout = new QHBoxLayout(selTab);
	horizontalLayout->addLayout(verticalLayout);
	scrollArea->setWidget(imagePlane);
	horizontalLayout->addWidget(scrollArea);
	imagePlane->update();
	return selTab;
}

void ld::createPopupMenu() {
	rect0Act = new QAction("A", this);
	rect1Act = new QAction("B", this);
	rect2Act = new QAction("C", this);
	rect3Act = new QAction("D", this);
	remAct   = new QAction("remove", this);

	popMenu = new QMenu();
	popMenu->addAction(rect0Act);
	popMenu->addAction(rect1Act);
	popMenu->addAction(rect2Act);
	popMenu->addAction(rect3Act);
	popMenu->addSeparator();
	popMenu->addAction(remAct);

	imagePlane->setPopMenu(popMenu);

	sigMap = new QSignalMapper(this);
	sigMap->setMapping(rect0Act, 0);
	sigMap->setMapping(rect1Act, 1);
	sigMap->setMapping(rect2Act, 2);
	sigMap->setMapping(rect3Act, 3);


	connect(rect0Act, SIGNAL(triggered()), sigMap, SLOT(map()));
	connect(rect1Act, SIGNAL(triggered()), sigMap, SLOT(map()));
	connect(rect2Act, SIGNAL(triggered()), sigMap, SLOT(map()));
	connect(rect3Act, SIGNAL(triggered()), sigMap, SLOT(map()));

	connect(sigMap, SIGNAL(mapped(int)), imagePlane, SLOT(addRectangle(int)));

	connect(remAct, SIGNAL(triggered()), imagePlane, SLOT(removeRectangle()));
}

void ld::startProcess() {
	qDebug() << "TODO: to be implemented !!! ";

	if (Image == NULL) {
		QMessageBox::information(this, tr("Image Viewer"),
								 tr("No image loaded!"));
		return;
	}

	// parameter zusammenfrickeln
	// FIXME threshold is hardcoded
	double edge_threshold = 0.2;
	ldc->set_edge_threshold(edge_threshold);
	int num_objects = 0;
	std::vector<std::vector<vole::Rectf> > object_rois;
	unsigned int number_rois = 0;
	std::vector<vole::Rectf> rois1, rois2, rois3, rois4;
	cv::Mat_<double> error_matrix;

	int count[4]; // count rois in objects a, b, c, d
	count[0] = count[1] = count[2] = count[3] = 0;

	QList<Rectangle*> &rectList = imagePlane->getRectList();
	number_rois = rectList.size();
	for (unsigned int i = 0; i < number_rois; ++i) {
		char tmp = rectList[i]->text().toAscii()[0];
		int obj_idx = static_cast<int>(tmp - 'A');
		std::cout << "object_idx = " << obj_idx << std::endl;
		count[obj_idx]++;
	}

	// create matrices
	count[0] = count[1] = count[2] = count[3] = 0;
	for (unsigned int i = 0; i < number_rois; ++i) {
		char tmp = rectList[i]->text().toAscii()[0];
		int obj_idx = static_cast<int>(tmp - 'A');
		QRect r = imagePlane->getRect(i);
		vole::Rectf rect(r.left(), r.top(), r.width(), r.height());
		if (obj_idx == 0) {
			rois1.push_back(rect);
			std::cout << "obj 0: " << r.left() << " " << r.top() << " " << r.right() << " " << r.bottom() << std::endl;
		}
		if (obj_idx == 1) {
			rois2.push_back(rect);
			std::cout << "obj 1: " << r.left() << " " << r.top() << " " << r.right() << " " << r.bottom() << std::endl;
		}
		if (obj_idx == 2) {
			rois3.push_back(rect);
			std::cout << "obj 2: " << r.left() << " " << r.top() << " " << r.right() << " " << r.bottom() << std::endl;
		}
		if (obj_idx == 3) {
			rois4.push_back(rect);
			std::cout << "obj 3: " << r.left() << " " << r.top() << " " << r.right() << " " << r.bottom() << std::endl;
		}
		count[obj_idx]++;
	}

	if (count[0] > 0) { object_rois.push_back(rois1); num_objects++; }
	if (count[1] > 0) { object_rois.push_back(rois2); num_objects++; }
	if (count[2] > 0) { object_rois.push_back(rois3); num_objects++; }
	if (count[3] > 0) { object_rois.push_back(rois4); num_objects++; }

	ldc->compute_lighting_direction(
		num_objects,
		object_rois,
		number_rois,
		error_matrix
	);
	std::cout << "error matrix: " << std::endl;
	for (int y = 0; y < num_objects; ++y) {
		for (int x = 0; x < num_objects; ++x) {
			std::cout << error_matrix[y][x] << " ";
		}
		std::cout << std::endl;
	}
}


