#include "viewerwindow.h"
//#include "view3d.h"
#include <qapplication.h>
#include <iostream>

int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <filename>\n\n"
					 "Filename may point to a RGB image or "
					 "a multispectral image descriptor file." << std::endl;
		return 1;
	}

	// load image   
	multi_img* image = new multi_img(argv[1]);
	if (image->empty())
		return 2;

	// start gui
	QApplication app(argc, argv);
	
	// regular viewer
	ViewerWindow window(image);
	window.show();
	
/*	// fancy 3d viewer
	View3D window3d(image);
	window3d.show();
*/	
	return app.exec();
}

