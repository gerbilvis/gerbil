#include "command_wrapper.h"
#include <QFileDialog>
#include <QMessageBox>
#include <iostream>
#include <fstream>

std::pair<QString, QImage> CommandWrapper::openImage() {
	std::pair<QString, QImage> ret;

	// open file dialog
	ret.first = QFileDialog::getOpenFileName(NULL, tr("Select an image ..."), "",
						  tr("Image Files (*.png *.jpg *.bmp *.gif *.pgm)"));

	// a valid filename provided?
	if (!ret.first.isEmpty()) {

		// trying to load the image file
		QImage image(ret.first);
		if (image.isNull()) {
			QMessageBox::warning(NULL, tr("Image Forensics Toolbox"),
					 tr("The image file %1 could not be read.").arg(ret.first));
		} else
			ret.second = image;
	}
	
	return ret;
}

/* from the qt examples: http://qt.nokia.com/doc/4.4/mainwindows-sdi-mainwindow-cpp.html */
QMainWindow *CommandWrapper::findMainWindow(const QString &window_title)
{
	foreach (QWidget *widget, qApp->topLevelWidgets()) {
		QMainWindow *mainWin = qobject_cast<QMainWindow *>(widget);
		if (mainWin && mainWin->windowTitle() == window_title)
			return mainWin;
	}
	return 0;
}

