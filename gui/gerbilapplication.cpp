
#include <dialogs/openrecent/openrecent.h>
#include <multi_img.h>
#include <controller/controller.h>
#include <app/gerbil_app_support.h>
#include "gerbilapplication.h"

#include <QFileInfo>
#include <QIcon>
#include <QStringList>
#include <QTextStream>

#include <cstdio>

GerbilApplication::GerbilApplication(int &argc, char **argv)
	: QApplication(argc, argv),
	  limitedMode(false)

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
	if (QIcon::themeName().isEmpty())
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
	Controller ctrl(imageFilename, limitedMode, labelsFilename);

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

