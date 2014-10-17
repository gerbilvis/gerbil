#ifndef OPENRECENT_H
#define OPENRECENT_H

#include <QDialog>
#include <QList>

#include "recentfile.h"

namespace Ui {
class OpenRecent;
}

class QStandardItemModel;
class QItemSelection;
class QModelIndex;

/** Dialog to open a gerbil image file with recent files list.
 *
 * Note: The recent files list is not updated by this class. The application
 * code needs to do this once a file was successfully loaded.
 *
 * See RecentFile::loadRecentFilesList(), RecentFile::saveRecentFilesList().
 */
class OpenRecent : public QDialog
{
	Q_OBJECT

public:

	explicit OpenRecent(QWidget *parent = 0);
	~OpenRecent();

	/** The file selected by the user for opening.
	 *
	 * If no file was selected or the file does not exist, an empty string is
	 * returned.
	 */
	QString getSelectedFile() const;

public slots:

	int exec ();

protected:

	void showEvent(QShowEvent * event);
	void keyPressEvent(QKeyEvent * event);

private slots:

	void browseForFile();
	void processAccepted();
	void processRejected();
	void processFileNameChanged(QString fileName);
	void processSelectionChanged(const QItemSelection & selected,
								 const QItemSelection & deselected);
	void processItemDoubleClicked(const QModelIndex & index);

private:

	void initRecentFilesUi();
	void browseForFileOnce();

private:

	Ui::OpenRecent *ui;

	// List of recently opened files. recentFilesList in QSettings.
	QList<RecentFile> recentFiles;
	// The last used path where an image was opened. recentPath in QSettings.
	QString recentPath;

	// item model for the list view
	QStandardItemModel *itemModel;

	// QDialog does not support exiting from showEvent() some reason. This is
	// the workaround.
	bool exitEarly;
};

#endif // OPENRECENT_H
