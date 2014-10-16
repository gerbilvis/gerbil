#include "recentfile.h"

#include <iostream>
#include <algorithm>

#include <QSettings>
#include <QFile>
#include <QFileInfo>
#include <QSet>


#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

QDataStream &operator<<(QDataStream &out, const RecentFile &rf)
{
	return out << rf.fileName
			   << rf.lastOpenenedTimestamp
			   << rf.previewImage;
}


QDataStream &operator>>(QDataStream &in, RecentFile &rf)
{
	return in  >> rf.fileName
			   >> rf.lastOpenenedTimestamp
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
	foreach (RecentFile const& rf, recentFiles) {
		if (! seen.contains(rf.fileName)) {
			tmp.append(rf);
			seen.insert(rf.fileName);
		}
	}
	recentFiles = tmp;
}


/** Loads the recent files list from QSettings. */
static QList<RecentFile> loadRecentFilesList()
{
	// app name is expected set in main.cpp
	QSettings settings;
	QList<RecentFile> recentFiles;
	int recentFilesVersion = settings.value("recentFilesVersion").toInt();
	if (recentFilesVersion < 1) {
		settings.remove("recentFilesList");
	} else {
		QList<QVariant> varl = settings.value("recentFilesList").toList();
		foreach (QVariant rf, varl) {
			recentFiles.append(rf.value<RecentFile>());
		}
		makeListEntriesUnique(recentFiles);
	}
	settings.setValue("recentFilesVersion", 1);
	return recentFiles;
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

void RecentFile::saveRecentFilesList(const QList<RecentFile> &recentFilesx)
{
	QList<RecentFile> recentFiles = recentFilesx;
	makeListEntriesUnique(recentFiles);
	if (recentFiles.size() > MaxRecentFiles) {
		recentFiles.erase(recentFiles.begin() + MaxRecentFiles,
						  recentFiles.end());
	}

	// app name is set in main.cpp
	QSettings settings;
	QVariantList varl;
	foreach(RecentFile const& rf, recentFiles) {
		varl << QVariant::fromValue(rf);
	}
	settings.setValue("recentFilesList", varl);
}

void RecentFile::appendToRecentFilesList(const QString &fileName,
										 const QImage &previewImage)
{
	QFileInfo fi(fileName);
	if ( ! fi.isFile() || ! fi.exists()) {
		std::cerr << "RecentFile::appendToRecentFilesList(): "
				  << "refusing to append non-existant file "
				  << fileName.toStdString()
				  << std::endl;
		return;
	}
	RecentFile rf;
	rf.fileName = fileName;
	rf.lastOpenenedTimestamp = QDateTime::currentDateTimeUtc();
	rf.previewImage = previewImage;

	QList<RecentFile> recentFiles = recentFilesList();
	recentFiles.prepend(rf);
	saveRecentFilesList(recentFiles);
}
