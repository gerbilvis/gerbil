#ifndef RESAMPLING_DETECTION_GUI_H
#define RESAMPLING_DETECTION_GUI_H

#include <QWidget>
#include <QScrollBar>
#include <QStatusBar>
#include <QLabel>
#include <QTabWidget>
#include <QAction>
#include <QPushButton>
#include <QComboBox>
#include <QMessageBox>
#include <QCloseEvent>
#include <QFileDialog>
#include <QResizeEvent>
#include <QSignalMapper>
#include <QLineEdit>

#include "command_wrapper.h"
#include "scrollarea.h"
#include "executionthread.h"

#define     MAX_RESULT_TABS    10

class resampling_detection_gui : public CommandWrapper
{
	Q_OBJECT

public slots:
	void startProcess();
	void clearPicture();
	void workStarted(void);
	void workStopped(QImage*);
	void closeEvent(QCloseEvent*);

public:
	resampling_detection_gui(VoleGui *parent, std::pair<QString, QImage*>);
	~resampling_detection_gui();

	static CommandWrapper* spawn(VoleGui *parent);
	
private:

// useful vars
	QStringList confNames, confFiles;

// gui elements
	bool   confFlag;
	int    actTabIx;
	QWidget*    resTabs[ MAX_RESULT_TABS ];
	int    numResTabs;
	QWidget     *selTab, *cfgTab;
	QComboBox	*selConfig;
	QPushButton *btnExecute, *btnZoomIn, *btnZoomOut, *btnClear;

	Rectangle *r1, *r2;
	ScrollArea *scaImg, *scaRes;

// working thread
	ExecutionThread ex;

//    bool    eventFilter( QObject*, QEvent*);
	void createGUI(QWidget*);
	QWidget* createSelectionTab(void);
	QWidget* createResultTab(QImage*);
	void createActions(void);
	void findConfFiles(const QString &path);
};

#endif // RESAMPLING_DETECTION_GUI_H
