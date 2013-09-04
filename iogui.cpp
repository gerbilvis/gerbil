#include "iogui.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <opencv2/highgui/highgui.hpp>

bool IOGui::selectFilename(bool writing, const QString &name)
{
	if (!name.isEmpty()) {
		filename = name;
		return true;
	}

	if (writing) {
		// there is no sense in providing name argument
		filename = QFileDialog::getSaveFileName(parent,
												QString("Save %1").arg(desc));

		// append .png for dumb/lazy users
		if (!filename.isEmpty()) {
			QFileInfo info(filename);
			if (info.suffix().isEmpty())
				filename.append(".png");
		}
	} else {
		filename = QFileDialog::getOpenFileName(parent,
												QString("Open %1").arg(desc));
	}

	return (!filename.isEmpty());
}

cv::Mat IOGui::readFile(int flags, int height, int width)
{
	assert(!filename.isEmpty());

	cv::Mat image;
	QString errorstr;
	try {
		image = cv::imread(filename.toStdString(), flags);
	} catch (cv::Exception &e) {
		errorstr = QString("The %1 could not be read.\nReason: %2"
		"\nSupported are all image formats readable by OpenCV.")
				.arg(shortdesc).arg(e.what());
	}
	if (image.empty()) {
		errorstr = QString("The %1 %2 could not be read."
		"\nSupported are all image formats readable by OpenCV.")
				.arg(shortdesc).arg(filename);
	}

	if (!errorstr.isEmpty()) {
		QMessageBox::critical(parent,
							  QString("Error Loading %1").arg(desc), errorstr);
		return cv::Mat();
	}

	if (height < 1 || width < 1) // no expected dimensionality
		return image;

	if (image.rows != height || image.cols != width) {
		QMessageBox::critical(parent, QString("%1 Mismatch").arg(desc),
				QString("The %1 has wrong proportions."
						"\nIt has to be of size %2x%3 for this image.")
						.arg(shortdesc).arg(width).arg(height));
		return cv::Mat();
	}
	return image;
}

cv::Mat IOGui::readFile(const QString &name, int flags, int height, int width)
{
	bool success = selectFilename(false, name);
	return (success ? readFile(flags, height, width) : cv::Mat());
}

void IOGui::writeFile(const cv::Mat &output)
{
	assert(!filename.isEmpty());

	QString errorstr;
	try {
		bool success = cv::imwrite(filename.toStdString(), output);
		if (!success)
			errorstr = QString("Could not write to %1!").arg(filename);
	} catch (cv::Exception &e) {
		errorstr = QString("Could not write to %1!\nReason: %2")
						   .arg(filename).arg(QString(e.what()));
	}

	if (!errorstr.isEmpty())
		QMessageBox::critical(parent,
							  QString("Error Writing %1").arg(desc), errorstr);
}

void IOGui::writeFile(const QString &name, const cv::Mat &output)
{
	bool success = selectFilename(true, name);
	if (success)
		writeFile(output);
}
