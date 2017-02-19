#include "recentfile.h"

#include <iostream>
#include <algorithm>

#include <QSettings>
#include <QFile>
#include <QFileInfo>
#include <QSet>
#include <QModelIndex>
#include <QPixmap>

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

QDataStream &operator<<(QDataStream &out, const RecentFile &rf)
{
	return out << rf.fileName
			   << rf.lastOpenedTimestamp
			   << rf.previewImage;
}


QDataStream &operator>>(QDataStream &in, RecentFile &rf)
{
	return in  >> rf.fileName
			   >> rf.lastOpenedTimestamp
			   >> rf.previewImage;
}


/** Makes each recent file entry in list unique by fileName.
 *
 * Note: This is not std::unique()!
 */
void makeListEntriesUnique(QList<RecentFile> &recentFiles)
{
	QSet<QString> seen;
	QList<RecentFile> tmp;
	for (auto rf : recentFiles) {
		if (!seen.contains(rf.fileName)) {
			tmp.append(rf);
			seen.insert(rf.fileName);
		}
	}
	recentFiles = tmp;
}


/** Loads the recent files list from QSettings. */
static QList<RecentFile> loadRecentFilesList()
{
	QList<RecentFile> files;
	QSettings settings;
	int recentFilesVersion = settings.value("recentFilesVersion").toInt();
	if (recentFilesVersion < 1)
		return files;

	QList<QVariant> varl = settings.value("recentFilesList").toList();
	for (auto varrf : varl) {
		RecentFile rf = varrf.value<RecentFile>();
		GGDBGP(rf.getFileNameWithoutPath().toStdString() << ": "
		                                                    "QImage::Format " << rf.previewImage.format() << endl);
		if (QFileInfo(rf.fileName).exists())
			files.append(rf);
	}
	makeListEntriesUnique(files);
	return files;
}

QString RecentFile::getFileNameWithoutPath() const
{
	return QFileInfo(fileName).fileName();
}

const QSize &RecentFile::iconSize() {
	static const QSize is(64, 64);
	return is;
}

QPixmap RecentFile::scaleToIconSize(const QPixmap &pm)
{
	QPixmap tmp = pm.scaled(RecentFile::iconSize(),
							Qt::KeepAspectRatio,
							Qt::SmoothTransformation);
	return tmp;
}

QPixmap RecentFile::getPreviewPixmap() const
{
	QPixmap pvpm = QPixmap::fromImage(this->previewImage);
	if (pvpm.rect().contains(
				QRect(QPoint(0,0), RecentFile::iconSize())))
	{
		// stored image is greater than icon size, rescale
		pvpm = RecentFile::scaleToIconSize(pvpm);
	}
	return pvpm;
}

QString RecentFile::lastOpenedString() const
{
	QDateTime localts = lastOpenedTimestamp.toLocalTime();
	return localts.toString();
}

QList<RecentFile> RecentFile::recentFilesList()
{
	static QList<RecentFile> recentFiles;
	static bool loaded = false;
	if (! loaded) {
		recentFiles = loadRecentFilesList();
	}
	// implicit sharing, copy is cheap
	return recentFiles;
}

void RecentFile::saveRecentFilesList(const QList<RecentFile> &recentFiles)
{
	QList<RecentFile> files = recentFiles;
	makeListEntriesUnique(files);
	if (files.size() > MaxRecentFiles) {
		files.erase(files.begin() + MaxRecentFiles, files.end());
	}

	QVariantList varl;
	for (auto rf : files) {
		varl << QVariant::fromValue(rf);
	}

	QSettings settings;
	settings.setValue("recentFilesList", varl);
	settings.setValue("recentFilesVersion", 1);
}

void RecentFile::appendToRecentFilesList(const QString &fileName,
										 const QPixmap &previewPixmap)
{
	QFileInfo fi(fileName);
	if ( ! fi.isFile() || ! fi.exists()) {
		std::cerr << "RecentFile::appendToRecentFilesList(): "
				  << "refusing to append non-existent file "
				  << fileName.toStdString()
				  << std::endl;
		return;
	}
	RecentFile rf;
	rf.fileName = fileName;
	rf.lastOpenedTimestamp = QDateTime::currentDateTimeUtc();
	QImage pvim = RecentFile::scaleToIconSize(previewPixmap).toImage();
	// keep config storage small
	pvim = pvim.convertToFormat(QImage::Format_RGB16);
	rf.previewImage = pvim;
	QList<RecentFile> recentFiles = recentFilesList();
	recentFiles.prepend(rf);
	saveRecentFilesList(recentFiles);
}

RecentFile recentFile(const QModelIndex &index)
{
	QVariant qvar = index.data(RecentFile::RecentFileDataRole);
	RecentFile rf = qvar.value<RecentFile>();
	return rf;
}
