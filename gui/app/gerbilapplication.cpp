
#include "gerbilapplication.h"
#include <dialogs/openrecent/openrecent.h>
#include <multi_img.h>
#include <controller/controller.h>
#include <widgets/mainwindow.h>

#include <QFileInfo>
#include <QIcon>
#include <QTimer>
#include <QStringList>
#include <QMessageBox>

#include <iostream>
#include <string>

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

int main(int argc, char **argv)
{
	exit(GerbilApplication(argc, argv).exec());
}

GerbilApplication::GerbilApplication(int &argc, char **argv)
    : QApplication(argc, argv),
      limitedMode(false),
      ctrl(nullptr)
{
	// set variables for QConfig use in application
	setOrganizationName("Gerbil");
	setOrganizationDomain("gerbilvis.org");
	setApplicationName("Gerbil");

	// postpone everything else until event loop is up
	QTimer::singleShot(0, this, SLOT(run()));
}

void GerbilApplication::run()
{
	check_system_requirements();

	init_qt();

	init_opencv();

#ifdef GERBIL_CUDA
	init_cuda();
#endif

	parse_args();

	// create controller
	ctrl = new Controller(imageFilename, limitedMode, labelsFilename, this);
}

void GerbilApplication::parse_args()
{
	if (arguments().size() < 2) {
		printUsage();
		OpenRecent *openRecentDlg = new OpenRecent();
		openRecentDlg->exec();
		// empty string if cancelled
		imageFilename = openRecentDlg->getSelectedFile();
		openRecentDlg->deleteLater();
	} else {
		imageFilename = arguments()[1];
	}

	if (imageFilename.isEmpty()) {
		// no window here, user already confirmed this in OpenRecent dialog.
		std::cerr << "No input file given." << std::endl;
		this->exit(1);
	}

	// FIXME: Use QString instead of std::string.
	// do a more complicated transformation to preserve non-ascii filenames
	std::string fn = imageFilename.toLocal8Bit().constData();
	// determine limited mode in a hackish way
	std::pair<std::vector<std::string>, std::vector<multi_img::BandDesc> >
			filelist = multi_img::parse_filelist(fn);
	limitedMode = determine_limited(filelist);

	// get optional labeling filename
	if (arguments().size() >= 3) {
		labelsFilename = arguments()[2];
	}
}

QString GerbilApplication::imagePath()
{
	if (imageFilename.isEmpty())
		return QString();
	else
		return QFileInfo(imageFilename).path();
}

void GerbilApplication::userError(QString msg)
{
	static const QString header =
		"Gerbil cannot continue."
		"<br/><br/>\n";
	criticalError(QString(header) + msg);
}

void GerbilApplication::internalError(QString msg)
{
	static const QString header =
		"Gerbil encountered an internal error that cannot "
		"be recovered. To help fixing this problem please collect "
		"information on your <br/>"
		"<ul>"
		"  <li>operating system</li>"
		"  <li>graphics hardware and drivers</li>"
		"</ul> <br/>"
		" and copy & paste "
		"the following error message and send everything to <br/>"
		"<a href=\"mailto:info@gerbilvis.org\">info@gerbilvis.org</a>.<br/>\n"
		"<br/>\n"
		"Thank you!<br/>\n"
		"<br/>\n"
		"Error:<br/>\n";
	criticalError(QString(header) + msg);
}

void GerbilApplication::criticalError(QString msg)
{
	// HTMLify quick and dirty
	if (!msg.contains("<br>", Qt::CaseInsensitive) &&
			!msg.contains("<br/>", Qt::CaseInsensitive)) {
		msg.replace("\n", "\n<br/>");
	}

	QMessageBox::critical(NULL,
						  "Gerbil Critical Error", msg, QMessageBox::Close);

	if (ctrl && ctrl->mainWindow()) {
		ctrl->mainWindow()->close();
	}
	this->exit(1);
}

void GerbilApplication::printUsage()
{
#ifdef __unix__
	std::cerr << "Usage: " << arguments()[0].toStdString()
		<< " <filename> [labeling file]"
		<< std::endl
		<< std::endl
	    << "Filename may point to any RGB or hyperspectral image file."
	    << std::endl
	    << "Labeling file is an RGB image with matching geometry."
	    << std::endl;
#endif
}

