#ifndef COMMAND_WRAPPER_H
#define COMMAND_WRAPPER_H

#include "vole_gui.h"
#include <QStatusBar>
#include <utility>
#include <boost/program_options.hpp>

#include <QPushButton>

class CommandWrapper : public QWidget
{
 	Q_OBJECT
 	
public:
	CommandWrapper(VoleGui *parent)
	 : QWidget(parent), parent(parent), 
	  statusBar(parent->statusBar()), container(parent->container())
	{
	}

	CommandWrapper(VoleGui *parent, std::pair<QString, QImage> img)
	 : QWidget(parent), parent(parent), imgName(img.first), Image(img.second),
	  statusBar(parent->statusBar()), container(parent->container())
	{
	}

	virtual ~CommandWrapper() {
	};

	static std::pair<QString, QImage> openImage();

	/* from the qt examples: http://qt.nokia.com/doc/4.4/mainwindows-sdi-mainwindow-cpp.html */
	QMainWindow *findMainWindow(const QString &window_title);

protected:

	VoleGui *parent;

	QString imgName;
	QImage Image;

	// short link to status bar and container
	QStatusBar *statusBar;
	QTabWidget *container;
};


#endif
