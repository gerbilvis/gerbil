/********************************************/
/*   +++ GUI for displaying edge maps +++   */
/********************************************/

#include "qcanny.h"

QCanny::QCanny(QWidget *parent, QString title)
	:QMainWindow(parent)
{
	setWindowTitle(title);
	createLayout();
}
	
QCanny::~QCanny(){}

void QCanny::createLayout()
{
	m_top = new QWidget(this);
	
	m_topLayout = new QVBoxLayout;
	m_top->setLayout(m_topLayout);

	setCentralWidget(m_top);

	m_dxDock = new ImageDockWidget( "DX", m_top);
	m_dxDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_dxDock);
	addDockWidget ( Qt::AllDockWidgetAreas, m_dxDock);

	m_dyDock = new ImageDockWidget( "DY", m_top);
	m_dyDock->setAllowedAreas(Qt::AllDockWidgetAreas);
	m_topLayout->addWidget(m_dyDock);
	addDockWidget ( Qt::AllDockWidgetAreas, m_dyDock);
// 	QDowckWidget *m_dyDock;
// 	QDowckWidget *m_cannyDock;
// 	QDowckWidget *m_rgbDock;
}






ImageDockWidget::ImageDockWidget(QString title , QWidget *parent )
	:QDockWidget(title,parent)
{
	createLayout();
}


void ImageDockWidget::createLayout()
{
	m_top = new QWidget(this);
	m_topLayout = new QVBoxLayout;
	setLayout(m_topLayout);
}



