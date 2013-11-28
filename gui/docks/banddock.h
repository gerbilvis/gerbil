#ifndef BANDDOCK_H
#define BANDDOCK_H

#include "model/representation.h"

#include "ui_banddock.h"
#include <opencv2/core/core.hpp>

class BandView;
class GraphSegWidget;

class BandDock : public QDockWidget, private Ui::BandDock
{
	Q_OBJECT
	
public:
	explicit BandDock(cv::Rect fullImgSize, QWidget *parent = 0);
	~BandDock();
	/** Returns the BandView. */
	// It is OK for the controller to access BandView directly. It is a
	// separate entity and not just a GUI element. This is cleaner than
	// duplicating the entire BandView interface in BandDock.
	BandView *bandView() { return bv; }
	GraphSegWidget *graphSegWidget() { return gs; }

	// get bandId of currently shown band
	representation::t getCurRepresentation() { return curRepr; }
	// get representation of currently shown band
	int getCurBandId() {return curBandId;}

signals:
	// request pixmap update as our current data became invalid
	void bandRequested(representation::t, int);
	/* The label being edited by the user has changed. */
	void currentLabelChanged(int);
	// FIXME short -> int (affects LabelModel)
	// GUI elements should use int consistently. Convert from int to short for
	// storage if necessary.
	void clearLabelRequested(short labelIdx);
	/** User requested a new label (markerSelector) */
	void newLabelRequested();

public slots:
	void loadSeeds();
	void graphSegModeToggled(bool enable);

	void changeBand(representation::t repr, int bandId,
					QPixmap band, QString desc);

	void processImageUpdate(representation::t);
	void processLabelingChange(const cv::Mat1s &labels,
							   const QVector<QColor> &colors,
							   bool colorsChanged);
	void processLabelingChange(const cv::Mat1s &labels,
							   const cv::Mat1b &mask);

protected slots:
	void clearLabel();
	void processMarkerSelectorIndexChanged(int index);

protected:
	// event filter to intercept enter()/leave() on our view
	bool eventFilter(QObject *obj, QEvent *event);

	void initUi();

	// local copies
	QVector<QColor> labelColors;
	cv::Rect fullImgSize;

	// representation and bandId of currently shown band
	representation::t curRepr;
	int curBandId;

	// our band view
	BandView *bv;
	// our widget for graph segmentation controls
	GraphSegWidget *gs;
};

#endif // BANDDOCK_H
