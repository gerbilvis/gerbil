#include "viewerwindow.h"
#include <qapplication.h>

int main(int argc, char **argv)
{
	if (argc < 2)
		return 1;

	// load image   
	multi_img image(argv[1]);
	if (image.empty())
		return 2;

	/* compute spectral gradient */
	// log image data
	multi_img log = image.clone();
	log.apply_logarithm();
	multi_img gradient = log.spec_gradient();

	// start gui
	QApplication app(argc, argv);
	ViewerWindow window(image, gradient);
	window.show();
	return app.exec();
}

