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
	void graphSegModeToggled(bool enable);
	/** The label being edited by the user has changed. */
	void currentLabelChanged(int);
	// FIXME short -> int (affects LabelModel)
	// GUI elements should use int consistently. Convert from int to short for
	// storage if necessary.
	void clearLabelRequested(short labelIdx);
	/** User requested a new label (markerSelector) */
	void newLabelRequested();
public slots:
	void changeBand(QPixmap band, QString desc);

	void processSeedingDone();
	void processLabelingChange(const cv::Mat1s &labels, const QVector<QColor> &colors, bool colorsChanged);
	void processLabelingChange(const cv::Mat1s &labels, const cv::Mat1b &mask);

protected slots:
	void clearLabelOrSeeds();
	void processMarkerSelectorIndexChanged(int idx);

protected:
	void initUi();
	// local copy
	QVector<QColor> labelColors;
};

#endif // BANDDOCK_H
