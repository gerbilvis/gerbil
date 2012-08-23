#include <QtGui/QApplication>
#include "vole_gui.h"

#include <iostream>

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);
	VoleGui w;
	w.show();
	return a.exec();
}
