#include "viewerwindow.h"
//#include "view3d.h"
#include <qapplication.h>
#include <qfiledialog.h>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{

	// start gui
	QApplication app(argc, argv);

	// get input file name
	std::string filename;
	if (argc != 2) {
#ifdef __unix__
		std::cerr << "Usage: " << argv[0] << " <filename>\n\n"
					 "Filename may point to a RGB image or "
					 "a multispectral image descriptor file." << std::endl;
#endif
		filename = QFileDialog::getOpenFileName
		           	(0, "Open Descriptor or Image File").toStdString();
	} else {
		filename = argv[1];
	}

	// load image   
	multi_img* image = new multi_img(filename);
	if (image->empty())
		return 2;
	
	// regular viewer
	ViewerWindow window(image);
	window.show();
	
/*	// fancy 3d viewer
	View3D window3d(image);
	window3d.show();
*/	
	return app.exec();
}

