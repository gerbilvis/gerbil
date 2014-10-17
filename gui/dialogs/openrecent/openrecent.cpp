#include "openrecent.h"
#include "ui_openrecent.h"
#include "recentfiledelegate.h"

#include <QSettings>
#include <QVariant>
#include <QDir>
#include <QString>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QStandardItemModel>
#include <QItemSelection>
#include <QModelIndex>
#include <QKeyEvent>

#include <iostream>

//#define GGDBG_MODULE
#include <gerbil_gui_debug.h>


OpenRecent::OpenRecent(QWidget *parent) :
    QDialog(parent),
	ui(new Ui::OpenRecent),
	itemModel(0),
	exitEarly(false)
{
	ui->setupUi(this);

	connect(ui->browseBtn, SIGNAL(clicked()),
			this, SLOT(browseForFile()));
	connect(this, SIGNAL(accepted()),
			this, SLOT(processAccepted()));
	connect(this, SIGNAL(rejected()),
			this, SLOT(processRejected()));

	connect(ui->fileLineEdit, SIGNAL(textChanged(QString)),
			this, SLOT(processFileNameChanged(QString)));

	// expecting app name is set in main.cpp
	QSettings settings;

	recentFiles = RecentFile::recentFilesList();

	// Use the last used directory for opening an image file or the user's
	// home as base path.
	recentPath = settings.value("recentPath",
								QDir::homePath()).toString();

	// Don't show recent files list if there aren't any.
	if (recentFiles.size() == 0) {
		ui->recentFilesListView->hide();
		ui->recentFilesListViewLabel->hide();
		this->resize(sizeHint().width(), minimumHeight());
		// trigger update
		ui->fileLineEdit->setText(QString());
	} else {
		initRecentFilesUi();
	}
}

OpenRecent::~OpenRecent()
{
	delete ui;
}

void OpenRecent::initRecentFilesUi()
{
	itemModel = new QStandardItemModel(this);

	foreach(RecentFile const& rf, recentFiles) {
		QList<QStandardItem*> items;
		QString fileName = QFileInfo(rf.fileName).fileName();
		QStandardItem *item = new QStandardItem(fileName);
		item->setData(QVariant::fromValue(rf), RecentFile::RecentFileDataRole);
		item->setData(QVariant::fromValue<QPixmap>(rf.getPreviewPixmap()),
						  Qt::DecorationRole);
		items.append(item);
		itemModel->appendRow(items);
	}

	ui->recentFilesListView->setModel(itemModel);
	ui->recentFilesListView->setDragDropMode(QAbstractItemView::NoDragDrop);
	ui->recentFilesListView->setAcceptDrops(false);
	ui->recentFilesListView->setSelectionMode(
				QAbstractItemView::SingleSelection);
	ui->recentFilesListView->setSelectionBehavior(
				QAbstractItemView::SelectItems);
	ui->recentFilesListView->setEditTriggers(
				QAbstractItemView::NoEditTriggers);
	ui->recentFilesListView->setHorizontalScrollBarPolicy(
				Qt::ScrollBarAlwaysOff);
	ui->recentFilesListView->setItemDelegate(new RecentFileDelegate(this));

	connect(ui->recentFilesListView->selectionModel(),
			SIGNAL(selectionChanged(QItemSelection,QItemSelection)),
			this,
			SLOT(processSelectionChanged(QItemSelection,QItemSelection)));
	connect(ui->recentFilesListView,
			SIGNAL(doubleClicked(QModelIndex)),
			this,
			SLOT(processItemDoubleClicked(QModelIndex)));

	// select the first item, if any
	if (recentFiles.size() > 0) {
		QModelIndex firstIdx = itemModel->index(0,0);
		ui->recentFilesListView->selectionModel()->select(
					QItemSelection(firstIdx,firstIdx),
					QItemSelectionModel::Select);
	}
}

void OpenRecent::browseForFileOnce()
{
	if (exitEarly) {
		return;
	}
	if (0 == recentFiles.size()) {
		exitEarly = true;
		setResult(Rejected);
		browseForFile();
		if (! getSelectedFile().isEmpty()) {
			setResult(Accepted);
		}
	}
}

QString OpenRecent::getSelectedFile() const
{
	QString filepath = ui->fileLineEdit->text();
	QFileInfo fi(filepath);
	if (filepath.isEmpty() || ! fi.isFile() || ! fi.exists()) {
		filepath = QString();
	}
	return filepath;
}

int OpenRecent::exec()
{
	browseForFileOnce();
	if (exitEarly) {
		return result();
	}
	return QDialog::exec();
}

void OpenRecent::showEvent(QShowEvent *event)
{
	QDialog::showEvent(event);
	browseForFileOnce();
}

void OpenRecent::keyPressEvent(QKeyEvent *event)
{
	// If up/down key pressed in line edit focus on listView and forward
	// event.
	if (! ui->recentFilesListView->hasFocus() &&
			(event->key() == Qt::Key_Up || event->key() == Qt::Key_Down))
	{
		GGDBGM("got QKeyEvent Up/Down" << endl);
		event->accept();
		ui->recentFilesListView->setFocus();
		// Is there any clever way to clone an event?
		QKeyEvent *evclone = new QKeyEvent(event->type(),
										   event->key(),
										   event->modifiers(),
										   event->text(),
										   event->isAutoRepeat(),
										   event->count());
		QApplication::postEvent(ui->recentFilesListView, evclone);
	} else {
		QDialog::keyPressEvent(event);
	}
}

void OpenRecent::browseForFile()
{
	// see QFileDialog::getOpenFileName()
	static const QString fileTypesString =
			"Images(*.png *.tif *.jpg *.txt);;"
			"All files (*)";
	QFileDialog::Options options;
	options &=  ! QFileDialog::HideNameFilterDetails;

	// Try to get path for open file dialog from line edit. If this does not
	// yield a valid path, try recentPath which is either the last used
	// directory or the user's home.
	QString dir = ui->fileLineEdit->text();
	QFileInfo fi(dir);
	if (dir.isEmpty()) {
		dir = recentPath;
	} else if (! fi.isDir()) {
		dir = fi.path();
		GGDBGM("not a dir, using path component " << dir.toStdString() << endl);
	}
	fi = QFileInfo(dir);
	GGDBGM("trying " << dir.toStdString() << endl);
	if (! fi.isDir() || ! QDir(dir).exists()) {
		GGDBGM("still not a dir, using recent path or home "
			   << recentPath.toStdString() << endl);
		dir = recentPath;
	}

	QString fileName = QFileDialog::getOpenFileName(
				this,
				"Open Image",
				dir,
				fileTypesString,
				0,
				options);
	GGDBGM("dialog returned \"" << fileName.toStdString() << "\"" << endl);
	if (! fileName.isEmpty()) {
		ui->fileLineEdit->setText(fileName);
		accept();
	}
	ui->fileLineEdit->setFocus();
}

void OpenRecent::processAccepted()
{
	// nothing here
}

void OpenRecent::processRejected()
{
	// make sure we return the empty string
	ui->fileLineEdit->setText(QString());
}

void OpenRecent::processFileNameChanged(QString fileName)
{
	QFileInfo file(fileName);
	ui->buttonBox->button(QDialogButtonBox::Open)->setEnabled(
				 file.isFile() && file.isReadable());
}

void OpenRecent::processSelectionChanged(const QItemSelection &selected,
										 const QItemSelection &deselected)
{
	QModelIndexList indexList = selected.indexes();
	if (selected.indexes().empty()) {
		return;
	}
	RecentFile rf = QVariant(selected.indexes().first().data(
				RecentFile::RecentFileDataRole)).value<RecentFile>();
	ui->fileLineEdit->setText(rf.fileName);
}

void OpenRecent::processItemDoubleClicked(const QModelIndex &index)
{
	const RecentFile rf = QVariant(index.data(
				RecentFile::RecentFileDataRole)).value<RecentFile>();
	ui->fileLineEdit->setText(rf.fileName);
	accept();
}
