#ifndef RECENTFILE_H
#define RECENTFILE_H

#include <QString>
#include <QDateTime>
#include <QImage>
#include <QVariant>
#include <QList>
#include <QPixmap>

class QModelIndex;
class QPixmap;

/** Manage recent file entries for opening gerbil image files.
 *
 * Note: Make sure QCoreApplication has the application name set before using
 * the member functions of RecentFile. This is usually done in main().
 */
struct RecentFile {
	// absolute filepath, unique key for RecentFile
	QString fileName;
	// timestamp in UTC
	QDateTime lastOpenedTimestamp;
	// 64x64 RGB preview image
	QImage previewImage;

	/** Qt::ItemDataRole for QStandardItem for storing RecentFile objects as
	 * QVariant. */
	enum { RecentFileDataRole  = Qt::UserRole + 1 };

	/** The maximum number of recent file entries stored in the recent files
	 * list.*/
	enum { MaxRecentFiles = 10 };

	bool operator==(const RecentFile &other) {
		return this->fileName == other.fileName;
	}

	QString getFileNameWithoutPath() const;

	/** Size of previewImage. */
	static QSize const& iconSize();

	/** Transform QPixmap by scaling to iconSize(). */
	static QPixmap scaleToIconSize(QPixmap const& pm);

	/** Returns the previewImage as QPixmap.
	 *
	 * If previewImage is larger than iconSize(), the pixmap is scaled
	 * proportionally.
	 */
	QPixmap getPreviewPixmap() const;

	/** Returns the last opened timestamp in local time formatted using the
	 * default localized format, e.g. "Wed May 20 03:40:13 1998". */
	QString lastOpenedString() const;

	/** Get the recent files list.
	 *
	 * The the recent files list is automatically loaded from QSettings.
	 * Modfications to the returned list are local (copy on write).
	 * To save the modified list use appendToRecentFilesList() or
	 * saveRecentFilesList().
	 */
	static QList<RecentFile> recentFilesList();

	/** Appends a filename to the recent files list, optionally accompanied by
	 * a preview image.
	 *
	 * This function also stores the updated recent files list in QSettings. The
	 * previewPixmap is automatically scaled down to RecentFile::iconSize().
	 */
	static void appendToRecentFilesList(QString const& fileName,
										const QPixmap &previewPixmap);

	/** Stores the recent files list in QSettings. */
	static void saveRecentFilesList(QList<RecentFile> const& recentFiles);
};

Q_DECLARE_METATYPE(RecentFile)

QDataStream &operator<<(QDataStream &out, const RecentFile &rf);
QDataStream &operator>>(QDataStream &in, RecentFile &rf);
inline uint qHash(RecentFile const& rf) {
	return qHash(rf.fileName);
}

RecentFile recentFile(const QModelIndex & index );

#endif // RECENTFILE_H

