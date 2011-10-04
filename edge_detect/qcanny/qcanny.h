/********************************************/
/*   +++ GUI for displaying edge maps +++   */
/********************************************/

#ifndef QCANNY_H
#define QCANNY_H

#include <iostream>

#include "myCanny.h"
#include "highgui.h"
#include "cv.h"

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QVBoxLayout>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QPixmap>
#include <QtGui/QSlider>
#include <QtGui/QLineEdit>
#include <QtGui/QDropEvent>
#include <QtGui/QDragEnterEvent>
#include <QtGui/QPushButton>
#include <QtCore/QMimeData>

class ImageDockWidget;

class QCanny : public QMainWindow
{
		Q_OBJECT

	signals:

	public slots:

	private slots:

	void updateLineEditLow(int);
	void updateLineEditHigh(int);

	void updateSliderLow(const QString);
	void updateSliderHigh(const QString);

	void updateEdge();
	void set();

	public:

		QCanny ( QWidget *parent, QString title );
		~QCanny();

	protected:

	private:
	
		cv::Mat dx, dy, gray, edge;
		void createLayout();


	public:

	protected:

	private:
		QWidget *m_top;
		QVBoxLayout *m_topLayout;
		QWidget *m_sliderWidget;
		QGridLayout *m_sliderLayout;

		ImageDockWidget *m_dxDock;
		ImageDockWidget *m_dyDock;
		ImageDockWidget *m_cannyDock;
		ImageDockWidget *m_rgbDock;
		
		QLabel *m_zeroLow;
		QLabel *m_zeroHigh;
		QLabel *m_fullLow;
		QLabel *m_fullHigh;

		QLineEdit *m_lValue;
		QLineEdit *m_hValue;
		QSlider *m_lTh;
		QSlider *m_hTh;

		QPushButton *m_setButton;

		myCanny can;

};

class ImageDockWidget : public QDockWidget
{
Q_OBJECT

public:

	ImageDockWidget(QString title,QString fn, QWidget *parent);
	~ImageDockWidget();

	QString path(){return m_path;}
	void setPixmap(QPixmap p){m_imgLabel->setPixmap(p);};

protected:
	void dropEvent ( QDropEvent * event );
	void dragEnterEvent ( QDragEnterEvent * event );

private:

	void createLayout();

private:

	QWidget *m_top;
	QVBoxLayout *m_topLayout;
	

	QLabel *m_imgLabel;
	QString m_path;

	QString m_initPath;

};

#endif
