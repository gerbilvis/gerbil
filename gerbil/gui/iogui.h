#ifndef IOGUI_H
#define IOGUI_H

#include <qtopencv.h>
#include <QString>

class IOGui
{
public:
	IOGui(const QString &descriptor, const QString &shortname, QWidget *owner=0)
		: desc(descriptor), shortdesc(shortname), parent(owner) {}

	bool selectFilename(bool writing, const QString& filename = QString());

	cv::Mat readFile(int flags = -1, int height = -1, int width = -1);
	// convenience overload that calls selectFilename first
	cv::Mat readFile(const QString& filename,
					 int flags = -1, int height = -1, int width = -1);

	void writeFile(const cv::Mat &output);
	// convenience overload that calls selectFilename first
	void writeFile(const QString& filename, const cv::Mat &output);

protected:
	QWidget *parent;
	QString desc, shortdesc, filename;
};

#endif // IOGUI_H
