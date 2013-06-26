#ifndef BANDDOCK_H
#define BANDDOCK_H

#include <QDockWidget>
#include "ui_banddock.h"


// FIXME button icons

class BandDock : public QDockWidget, private Ui::BandDock
{
	Q_OBJECT
	
public:
	explicit BandDock(QWidget *parent = 0);
	~BandDock();
	/** Returns the BandView. */
	// It is OK for the controller to access BandView directly. It is a
	// separate entity and not just a GUI element. This is cleaner than
	// duplicating the entire BandView interface in BandDock.
	BandView *bandView() {return bv;}

signals:
	void graphSegRequested();
public slots:
	void changeBand(QPixmap band, QString desc);
	//void changeCurrentLabel(int);
	//void toggleShowLabels(bool);
	//void toggleSingleLabel(bool);
	//void applyLabelAlpha(int);
	//void clearLabelOrSeeds();

protected:
	void initUi();

};

#endif // BANDDOCK_H
