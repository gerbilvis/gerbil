/********************************************/
/*   +++ GUI for displaying edge maps +++   */
/********************************************/

#include "qcanny.h"


QCanny::QCanny(QWidget *parent, QString title)
	:QMainWindow(parent)
{
	setWindowTitle(title);
	createLayout();
	setAcceptDrops(true);
}
	
QCanny::~QCanny(){}

void QCanny::createLayout()
{
	m_top = new QWidget(this);
	
	m_topLayout = new QVBoxLayout;
	m_top->setLayout(m_topLayout);
	setCentralWidget(m_top);

	m_zeroLow = new QLabel("0",m_top);
	m_zeroHigh = new QLabel("0",m_top);
	m_fullLow = new QLabel("100",m_top);
	m_fullHigh = new QLabel("100",m_top);

	m_sliderWidget = new QWidget(m_top);
	m_sliderLayout = new QGridLayout;
	m_sliderWidget->setLayout(m_sliderLayout);
	m_topLayout->addWidget(m_sliderWidget);




	m_lValue= new QLineEdit(m_top);
	m_lValue->setText("1");
	connect(m_lValue,SIGNAL(textChanged ( const QString )), this, SLOT(updateSliderLow(const QString)));
	m_sliderLayout->addWidget(m_lValue,0,0);
	m_hValue= new QLineEdit(m_top);
	m_hValue->setText("1");
	connect(m_hValue,SIGNAL(textChanged ( const QString )), this, SLOT(updateSliderHigh(const QString)));
	m_sliderLayout->addWidget(m_hValue,0,1);

	m_sliderLayout->addWidget(m_fullLow,1,0);
	m_sliderLayout->addWidget(m_fullHigh,1,1);
		
	m_lTh = new QSlider(Qt::Vertical,m_top);
	m_lTh->setTickInterval(1);
	m_lTh->setRange(1,100);
	m_lTh->setValue(1);
	m_lTh->setTickPosition(QSlider::TicksLeft);
	connect(m_lTh, SIGNAL(valueChanged(int)),this, SLOT(updateLineEditLow(int)));
	connect(m_lTh, SIGNAL(valueChanged(int)),this, SLOT(updateEdge()));
	m_sliderLayout->addWidget(m_lTh,2,0);
	

	m_hTh = new QSlider(Qt::Vertical,m_top);
	m_hTh->setTickInterval(1);
	m_hTh->setRange(1,100);
	m_hTh->setValue(1);
	m_hTh->setTickPosition(QSlider::TicksRight);
	connect(m_hTh, SIGNAL(valueChanged(int)),this, SLOT(updateLineEditHigh(int)));
	connect(m_hTh, SIGNAL(valueChanged(int)),this, SLOT(updateEdge()));
	m_sliderLayout->addWidget(m_hTh,2,1);

	m_sliderLayout->addWidget(m_zeroLow,3,0);
	m_sliderLayout->addWidget(m_zeroHigh,3,1);

	m_setButton = new QPushButton("Set",m_top);
	connect(m_setButton,SIGNAL(clicked()),this, SLOT(set()));
	m_sliderLayout->addWidget(m_setButton,4,0,1,2);


	m_dxDock = new ImageDockWidget( "DX",":/res/smile_dx.xpm", m_top);
	m_dxDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_dxDock);
	addDockWidget ( Qt::LeftDockWidgetArea, m_dxDock);

	m_dyDock = new ImageDockWidget( "DY",":/res/smile_dy.xpm", m_top);
	m_dyDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_dyDock);
	addDockWidget ( Qt::LeftDockWidgetArea, m_dyDock);

	m_cannyDock = new ImageDockWidget( "Canny",":/res/smile_edge.xpm", m_top);
	m_cannyDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_cannyDock);
	addDockWidget ( Qt::RightDockWidgetArea, m_cannyDock);

	m_rgbDock = new ImageDockWidget( "RGB",":/res/smile.xpm", m_top);
	m_rgbDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_rgbDock);
	addDockWidget ( Qt::RightDockWidgetArea, m_rgbDock);

}


void QCanny::updateLineEditHigh(int val)
{
	m_hValue->setText(QString().setNum(val));
}

void QCanny::updateLineEditLow(int val)
{
	m_lValue->setText(QString().setNum(val));
}

void QCanny::updateSliderLow(const QString val)
{
	m_lTh->setValue(val.toInt());
}
void QCanny::updateSliderHigh(const QString val)
{
	m_hTh->setValue(val.toInt());
	
}

void QCanny::updateEdge()
{
	can.canny(dx, dy, edge, m_lTh->value(), m_hTh->value(), 3, true);
	cv::Mat edgeShow;
	edge.convertTo( edgeShow, CV_8UC1, 255.);

	cv::imwrite("./edge.png", edge);
	
	QImage tmp("./edge.png");
	if (tmp.isNull()) 
	{
	  return;
  }


  m_cannyDock->setPixmap( QPixmap::fromImage( tmp ) );

}

void QCanny::set()
{
// 	cv::Mat dx, dy, gray, edge, cedge;
	
	const char* dxPath = m_dxDock->path().toAscii();
	const char* dyPath = m_dyDock->path().toAscii();
	dx = cv::imread(dxPath,0);
	dy = cv::imread(dyPath,0);
	cv::Mat dx_u;
	cv::Mat dy_u;

	dx.convertTo( dx_u, CV_8UC1, 255.);
	dy.convertTo( dy_u, CV_8UC1, 255.);

	edge = cv::Mat::zeros(dx.rows,dx.cols,CV_8UC1);

	
	can.canny(dx_u, dy_u, edge, m_lTh->value(), m_hTh->value(), 3, true);
	
	cv::Mat edgeShow;
	edge.convertTo( edgeShow, CV_8UC1, 255.);
	
	cv::imwrite("./edge.png", edgeShow);
	
	QImage tmp("./edge.png");
	if (tmp.isNull()) 
	{
	  return;
  }
	
  m_cannyDock->setPixmap( QPixmap::fromImage( tmp ));

}


ImageDockWidget::ImageDockWidget(QString title , QString fn, QWidget *parent )
	:QDockWidget(title,parent), m_initPath(fn)
{
	createLayout();
	setAcceptDrops(true);
}


void ImageDockWidget::createLayout()
{
	
	m_top = new QWidget(this);
	m_topLayout = new QVBoxLayout;
	m_top->setLayout(m_topLayout);
	setWidget(m_top);

	QPixmap pm = QPixmap(m_initPath);
	m_imgLabel = new QLabel(m_top);
	m_imgLabel->setPixmap(pm);

	m_topLayout->addWidget(m_imgLabel);
	

}

void ImageDockWidget::dragEnterEvent(QDragEnterEvent *event)
{
    setBackgroundRole(QPalette::Highlight);
    event->acceptProposedAction();
//     emit changed(event->mimeData());
}

void ImageDockWidget::dropEvent ( QDropEvent * event )
{
	const QMimeData *data = event->mimeData();

	if(data->hasUrls())
	{
		m_path = data->text();
		m_path.remove("file://");
		QPixmap p;
		p.load(m_path);
		m_imgLabel->setPixmap(p);

		
		event->acceptProposedAction();

	}

}

ImageDockWidget::~ImageDockWidget(){}


