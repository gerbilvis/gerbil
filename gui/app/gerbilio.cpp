#include "gerbilio.h"
#include "gerbilapplication.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <opencv2/highgui/highgui.hpp>

boost::shared_ptr<QMap<QString, QString> > GerbilIO::lastUsedDirMap;

GerbilIO::GerbilIO(QWidget *parent, const QString &description, const QString &descShort)
	: parent(parent),
	  description(description),
	  shortDesc(descShort),
	  filename(QString()), // null
	  category(QString()), // null
	  cvflags(-1),
	  width(0),
	  height(0)
{
}

void GerbilIO::setFileName(QString filename)
{
	this->filename = filename;
}

void GerbilIO::setFileSuffix(QString suffix)
{
	if (!suffix.startsWith(".") && suffix.length() > 0)
		suffix = QString(".") + suffix;
	this->suffix = suffix;
}

void GerbilIO::setFileCategory(QString const& cat)
{
	this->category = cat;
}

void GerbilIO::setOpenCVFlags(int cvflags)
{
	this->cvflags = cvflags;
}

cv::Mat GerbilIO::readImage()
{
	if (filename.isEmpty())
		selectFileName(SelectForReading);
	if (filename.isEmpty())
		return cv::Mat();

	cv::Mat image;
	QString errorstr;
	try {
		image = cv::imread(filename.toStdString(), cvflags);
	} catch (cv::Exception &e) {
		errorstr = QString(
					"The %1 could not be read.<br/>"
					"Reason: %2<br/>"
					"Supported are all image formats readable by OpenCV.")
				.arg(shortDesc).arg(e.what());
	}
	if (image.empty()) {
		errorstr = QString(
					"The %1 \"%2\" could not be read.<br/>"
					"Supported are all image formats readable by OpenCV.")
				.arg(shortDesc).arg(filename);
	}
	if (!errorstr.isEmpty()) {
		QMessageBox::critical(parent,
							  QString("Error Loading %1").arg(description), errorstr);
		return cv::Mat();
	}

	if (height < 1 || width < 1) // no expected dimensionality
		return image;

	if (image.rows != height || image.cols != width) {
		QMessageBox::critical(parent, QString("%1 Mismatch").arg(description),
				QString("The %1 has wrong proportions.<br/>"
						"It has to be of size %2x%3 for this image.")
						.arg(shortDesc).arg(width).arg(height));
		return cv::Mat();
	}
	return image;
}

void GerbilIO::writeImage(const cv::Mat &mat)
{
	if (filename.isEmpty())
		selectFileName(SelectForWriting);
	if (filename.isEmpty())
		return;

	QString errorstr;
	try {
		if (!cv::imwrite(filename.toStdString(), mat))
			errorstr = QString("Could not write to %1!").arg(filename);
	} catch (cv::Exception &e) {
		errorstr = QString("Could not write file \"%1\"!<br/>"
						   "Reason: %2")
				.arg(filename).arg(QString(e.what()));
	}

	if (!errorstr.isEmpty())
		QMessageBox::critical(parent,
							  QString("Error Writing %1").arg(description), errorstr);

}

void GerbilIO::selectFileName(SelectMode mode)
{
	QString fileDialogDir = lastUsedDir(category);
	if (fileDialogDir.isEmpty() && GerbilApplication::instance()) {
		// use image path if no last used dir
		fileDialogDir = GerbilApplication::instance()->imagePath();
	}

	if (SelectForWriting == mode) {
		QString caption = QString("Save %1").arg(description);
		filename = QFileDialog::getSaveFileName(
					parent,
					caption,
					fileDialogDir
					);

		// append suffix if no suffix
		if (!filename.isEmpty() && QFileInfo(filename).suffix().isEmpty())
			filename.append(suffix);

	} else { // SelectForReading == mode
		// TODO Add support for file type.
		QString caption = QString("Open %1").arg(description);
		filename = QFileDialog::getOpenFileName(parent,
												caption,
												fileDialogDir
												);
	}

	QFileInfo fileInfo(filename);
	if (!filename.isEmpty() && !fileInfo.filePath().isEmpty()) {
		setLastUsedDir(category, fileInfo.filePath());
	}
}

void GerbilIO::setLastUsedDir(QString cat, QString path)
{
	if (cat.isEmpty())
		return;
	if (!lastUsedDirMap) {
		lastUsedDirMap = boost::shared_ptr<QMap<QString, QString> >(
					new QMap<QString, QString>());
	}
	lastUsedDirMap->insert(cat, path);
}

QString GerbilIO::lastUsedDir(QString cat)
{
	if (!cat.isEmpty() && lastUsedDirMap)
		return lastUsedDirMap->value(cat);
	else
		return QString();
}
QString GerbilIO::getFilename() const
{
	return filename;
}

void GerbilIO::setHeight(int value)
{
	if (value >= 0)
		height = value;
}

void GerbilIO::setWidth(int value)
{
	if (value >= 0)
		width = value;
}
