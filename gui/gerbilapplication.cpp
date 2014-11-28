
#include <dialogs/openrecent/openrecent.h>
#include <multi_img.h>
#include <controller/controller.h>
#include <app/gerbil_app_support.h>
#include "gerbilapplication.h"

#include <QIcon>
#include <QStringList>
#include <QTextStream>

#include <cstdio>

GerbilApplication::GerbilApplication(int &argc, char **argv)
	:QApplication(argc, argv)
{
}

GerbilApplication::GerbilApplication(int &argc, char **argv, bool GUIenabled)
	:QApplication(argc, argv, GUIenabled)
{
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

	// get input file name
	QString filename;
	if (arguments().size() < 2) {
#ifdef __unix__
		err << "Usage: " << arguments()[0]
			<< " <filename> [labeling file]"
			<< endl
			<< endl
			<< "Filename may point to a RGB image or "
			<< "a multispectral image descriptor file."
			<< endl;
#endif
		OpenRecent *openRecentDlg = new OpenRecent();
		openRecentDlg->exec();
		// empty string if cancelled
		filename = openRecentDlg->getSelectedFile();
		openRecentDlg->deleteLater();
	} else {
		filename = arguments()[1];
	}

	if (filename.isEmpty()) {
		// no window here, user already confirmed this in OpenRecent dialog.
		err << "No input file given." << endl;
		exit(ExitNoInput);
	}

	// FIXME: Use QString instead of std::string.
	// do a more complicated transformation to preserve non-ascii filenames
	std::string fn = filename.toLocal8Bit().constData();
	// determine limited mode in a hackish way
	std::pair<std::vector<std::string>, std::vector<multi_img::BandDesc> >
			filelist = multi_img::parse_filelist(fn);
	bool limited_mode = determine_limited(filelist);

	// get optional labeling filename
	QString labelfile;
	if (arguments().size() >= 3) {
		labelfile = arguments()[2];
	}

	// create controller
	Controller ctrl(filename, limited_mode, labelfile);

	// run Qt event loop
	exit(QApplication::exec());
}

