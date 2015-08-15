
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
	return GerbilApplication(argc, argv).exec();
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
		throw shutdown_exception();
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

void GerbilApplication::internalError(QString msg, bool critical)
{
	static const QString header =
	        "Gerbil encountered an internal error. "
	        "To help fixing this problem please send us a report with:"
	        "<ul>"
	        "  <li>steps you performed that lead up to this error</li>"
	        "  <li>the error message below</li>"
	        "  <li>your operating system</li>"
	        "  <li>vendor/model of your graphics hardware</li>"
	        "</ul>"
		"You can reach us via email at: "
		"<a href=\"mailto:info@gerbilvis.org\">info@gerbilvis.org</a>.<br/>\n"
		"<br/>\n"
		"Error message:<br/>\n";
	criticalError(QString(header) + msg, critical);
}

void GerbilApplication::criticalError(QString msg, bool quit)
{
	// HTMLify quick and dirty
	if (!msg.contains("<br>", Qt::CaseInsensitive) &&
			!msg.contains("<br/>", Qt::CaseInsensitive)) {
		msg.replace("\n", "\n<br/>");
	}

	if (quit) {
		QMessageBox::critical(NULL, "Gerbil Critical Error", msg, "Quit");
		auto app = instance();
		if (app && app->ctrl && app->ctrl->mainWindow()) {
			app->ctrl->mainWindow()->close();
		}
		GerbilApplication::exit(1);
		throw shutdown_exception();
	} else {
		QMessageBox::critical(NULL,
		                      "Gerbil Critical Error", msg, QMessageBox::Close);
	}
}

bool GerbilApplication::notify(QObject* receiver, QEvent* event) {
	try {
		return QApplication::notify(receiver, event);
	} catch (shutdown_exception &) {
		// do nothing, let Qt clean up after this
	} catch (std::exception&) {
		try {
			handle_exception(std::current_exception(), true);
		} catch (shutdown_exception &) {
			// gently eat up redundant throw
		}
	}
	return false;
}

void GerbilApplication::handle_exception(std::exception_ptr e, bool critical)
{
	try {
		if (e)
			std::rethrow_exception(e);
	} catch(const std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		/* Note: After an exception is thrown, the connection to the windowing
		 * server might already be closed. It is not safe to call a GUI related
		 * function after catching an exception. */
		internalError(e.what(), critical);
	}
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

