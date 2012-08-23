/********************************************/
/*   +++ GUI for displaying edge maps +++   */
/********************************************/

#include "qcanny.h"
#include <QtGui/QApplication>
 
int main( int argc, char* argv[])
{
	QApplication a(argc, argv);
	QCanny c(0, "QCanny Edge Viewer");
	c.show();
	return a.exec();
}
