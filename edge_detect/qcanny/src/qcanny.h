/********************************************/
/*   +++ GUI for displaying edge maps +++   */
/********************************************/

#ifndef QCANNY_H
#define QCANNY_H

#include <QtGui/QDockWidget>
#include <QtGui/QMainWindow>
#include <QtGui/QVBoxLayout>
#include <QtGui/QLabel>

class ImageDockWidget;

class QCanny : public QMainWindow
{
		Q_OBJECT

	signals:

	public slots:

	private slots:

	public:

		QCanny ( QWidget *parent, QString title );
		~QCanny();

	protected:

	private:

		void createLayout();


	public:

	protected:

	private:
		QWidget *m_top;
		QVBoxLayout *m_topLayout;

		ImageDockWidget *m_dxDock;
		ImageDockWidget *m_dyDock;
		ImageDockWidget *m_cannyDock;
		ImageDockWidget *m_rgbDock;

};

class ImageDockWidget : public QDockWidget
{
Q_OBJECT

public:

	ImageDockWidget(QString title,QWidget *parent);
	~ImageDockWidget();

private:

	void createLayout();

private:

	QWidget *m_top;
	QVBoxLayout *m_topLayout;


};

#endif
