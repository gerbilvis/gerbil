#ifndef GERBILIO_H
#define GERBILIO_H

#include <QString>
#include <QMap>

#include <boost/shared_ptr.hpp>

#include <qtopencv.h>

class QWidget;

/** Class for managing common user I/O interaction when loading / saving files.
 *
 * GerbilIO automatically queries the user for a file for reading/writing
 * unless a filename has been set programmatically. See setFileName().
 *
 * For file dialogs, GerbilIO manages a last used directory list for the
 * runtime of the application, one directory for each file category. Once a
 * file dialog has succeeded in saving/loading a file for a certain category,
 * the next time GerbilIO opens a dialog for this category, it will start in
 * the same directory. For the first dialog of each category opened during the
 * runtime of the applicaton, the path of the image file loaded by gerbil is
 * used.
 *
 * Typically a GerbilIO object is created, options set and a read or write
 * operation is executed.
 *
 * Examples
 *
 * Writing:
 *
 *    GerbilIO io(this, "Screenshot File", "screenshot");
 *    io.setFileSuffix(".png");
 *    io.setDirectoryCategory("Screenshot");
 *    io.writeImage(output);
 *
 * Reading:
 *    GerbilIO io(this, "Seed Image File", "seed image");
 *    io.setDirectoryCategory("SeedFile");
 *    io.setOpenCVFlags(0);
 *    io.setWidth(fullImgSize.width);
 *    io.setHeight(fullImgSize.height);
 *    cv::Mat1s seeding = io.readImage();
 *    if (seeding.empty())
 *       return;
 */
class GerbilIO
{
public:

	enum SelectMode {
		SelectForReading,
		SelectForWriting
	};

	/**
	 * @brief GerbilIO
	 * @param parent Parent widget.
	 * @param description Long description.
	 *	                  Used in dialog caption, e.g.
	 *                    "Open/Save <description>".
	 * @param descShort Short description.
	 *                  Used in dialog text, e.g.
	 *                  "Loading <shortDesc> failed".
	 */
	GerbilIO(QWidget *parent,
			 QString const& description,
			 QString const& shortDesc);

	/** Set filename used for reading / writing.
	 *
	 * By default no filename is set and the user will be queried using
	 * selectFileName().
	 */
	void setFileName(QString filename);

	QString getFilename() const;

	/** Set filename for reading/writing querying the user with a file dialog.
	 *
	 * description and shortDesc from the constructor are used here.
	 */
	void selectFileName(SelectMode mode);

	/** Set the filename suffix to use for writing.
	 *
	 * Suffix can be of the "png" or ".png".
	 * The suffix is only appended if the filename does not already have a
	 * suffix. By default no suffix is set.
	 */
	void setFileSuffix(QString suffix);

	/** Set category for remembering last used directory.
	 *
	 * By default NoCategory is set and the directory will not be remembered.
	 */
	void setFileCategory(const QString &cat);

	/** Set flags for cv::imread().
	 *
	 * By default flags=-1 is used for cv::imread().
	 */
	void setOpenCVFlags(int cvflags);

	/** Set the required width for the loaded image file.
	 *
	 * If the image does not have the correct size in readImage() an error
	 * dialog is shown and an empty image is returned.
	 */
	void setWidth(int value);

	/** Set the required height for the loaded image file.
	 *
	 * If the image does not have the correct size in readImage() an error
	 * dialog is shown and an empty image is returned.
	 */
	void setHeight(int value);

	/** Read image file with cv::imread().
	 *
	 * See setFileName(), setDirectoryCategory() and setOpenCVFlags() for
	 * setting options prior to reading.
	 */
	cv::Mat readImage();

	/** Write image data to file.
	 *
	 * See setFileName(), setDirectoryCategory() and setOpenCVFlags() for
	 * setting options prior to writing.
	 */
	void writeImage(const cv::Mat &mat);

private:

	void setLastUsedDir(QString cat, QString path);
	QString lastUsedDir(QString cat);

	QWidget *parent;
	QString description;
	QString shortDesc;
	QString filename;

	QString suffix;
	QString category;
	int cvflags;
	int width;
	int height;

	static boost::shared_ptr<QMap<QString, QString> > lastUsedDirMap;
};
#endif // GERBILIO_H
