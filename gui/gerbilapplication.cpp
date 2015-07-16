
#include "gerbilapplication.h"
#include <dialogs/openrecent/openrecent.h>
#include <multi_img.h>
#include <controller/controller.h>
#include <widgets/mainwindow.h>
#include <app/gerbil_app_support.h>

#include <QFileInfo>
#include <QIcon>
#include <QStringList>
#include <QTextStream>
#include <QMessageBox>

#include <cstdio>
#include <cstdlib>

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>


GerbilApplication::GerbilApplication(int &argc, char **argv)
    : QApplication(argc, argv),
      limitedMode(false),
      eventLoopStartedEvent(QEvent::registerEventType()),
      eventLoopStarted(false),
      ctrl(nullptr)
{
}

GerbilApplication *GerbilApplication::instance()
{
	return dynamic_cast<GerbilApplication*>(QCoreApplication::instance());
}

void GerbilApplication::run()
{
	QTextStream out(stdout);
	QTextStream err(stderr);

	registerQMetaTypes();

	// setup our custom icon theme if there is no system theme (OS X, Windows)
	if (QIcon::themeName().isEmpty() || !QIcon::themeName().compare("hicolor"))
		QIcon::setThemeName("Gerbil");

	if (!check_system_requirements()) {
		// TODO: window
		err << "Unfortunately the machine does not meet minimal "
			<< "requirements to launch Gerbil." << endl;
		exit(ExitSystemRequirements);
	}

	init_opencv();
	init_cuda();

	loadInput();

	// create controller
	ctrl = new Controller(imageFilename, limitedMode, labelsFilename, this);

	// get notified when the event-loop has fired up, see eventFilter()
	installEventFilter(this);
	postEvent(this, new QEvent(
				  QEvent::Type(eventLoopStartedEvent)));

	// run Qt event loop
	exit(QApplication::exec());
}

QString GerbilApplication::imagePath()
{
	if (imageFilename.isEmpty())
		return QString();
	else
		return QFileInfo(imageFilename).path();
}

void GerbilApplication::criticalError(QString msg)
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

	// HTMLify quick and dirty
	if (!msg.contains("<br>", Qt::CaseInsensitive) &&
			!msg.contains("<br/>", Qt::CaseInsensitive)) {
		msg.replace("\n", "\n<br/>");
	}

	QMessageBox::critical(NULL,
						  "Gerbil Critical Error",
						  QString(header) + msg,
						  QMessageBox::Close);

	if (eventLoopStarted) {
		GGDBGM("using GerbilApplication::exit()" << endl);
		if (ctrl && ctrl->mainWindow()) {
			ctrl->mainWindow()->close();
		}
		GerbilApplication::exit(GerbilApplication::ExitFailure);
	} else {
		GGDBGM("using std::exit()" << endl);
		std::exit(ExitFailure);
	}

}

bool GerbilApplication::eventFilter(QObject *obj, QEvent *event)
{
	if (event->type() == eventLoopStartedEvent) {
		GGDBGM("eventLoopStartedEvent captured" << endl);
		event->accept();
		eventLoopStarted = true;
		removeEventFilter(this);
		return true;
	} else {
		return QApplication::eventFilter(obj, event);
	}
}

void GerbilApplication::loadInput()
{
	QTextStream err(stderr);

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
		err << "No input file given." << endl;
		exit(ExitNoInput);
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

void GerbilApplication::printUsage()
{
#ifdef __unix__
	QTextStream err(stderr);
	err << "Usage: " << arguments()[0]
		<< " <filename> [labeling file]"
		<< endl
		<< endl
		<< "Filename may point to a RGB image or "
		<< "a multispectral image descriptor file."
		<< endl;
#endif
}

