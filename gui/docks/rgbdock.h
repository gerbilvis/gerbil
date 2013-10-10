#ifndef RGBDOCK_H
#define RGBDOCK_H

#include <QDockWidget>
#include <shared_data.h>
#include "../model/representation.h"
#include "../model/falsecolormodel.h"
#include "ui_rgbdock.h"

class ScaledView;


struct RgbDockState {
	enum Type {FINISHED=0, CALCULATING, ABORTING};
};
std::ostream &operator<<(std::ostream& os, const RgbDockState::Type& state);

class RgbDock : public QDockWidget, private Ui::RgbDock{
	Q_OBJECT
public:
	explicit RgbDock(QWidget *parent = 0);
	
signals:
	/** Request a rendering for coloringType of the current image and ROI.
	 *
	 * @param recalc If set, the result shall be recalculated wether or not an
	 *               up-to-date cached copy is available. Useful to request a
	 *               new SOM.
	 */
	void falseColoringRequested(FalseColoring::Type coloringType, bool recalc = false);
	/** Requests the model to cancel the previously requested calculation for
	 * coloringType */
	void cancelComputationRequested(FalseColoring::Type coloringType);
public slots:
	void processVisibilityChanged(bool visible);
	// from model: our displayed coloringType may be out of date
	void processColoringOutOfDate(FalseColoring::Type coloringType);
	void processCalculationProgressChanged(FalseColoring::Type coloringType, int percent);
	void processColoringComputed(FalseColoring::Type coloringType, QPixmap p);
	void processComputationCancelled(FalseColoring::Type coloringType);
protected slots:
	void processSelectedColoring(); // the selection in the combo box changed
	void processApplyClicked();
protected:
	void initUi();

	// the coloringType currently selected in the comboBox
	FalseColoring::Type selectedColoring();

	void requestColoring(FalseColoring::Type coloringType, bool recalc = false);
	// show/hide the progress bar depending on coloringState and update the value
	void updateProgressBar();
	// update/show/hide apply/cancel button according to state
	void updateTheButton();

	void enterState(FalseColoring::Type coloringType, RgbDockState::Type state);

	// True if the dock is visible (that is tab is selected or top level).
	// Note: This is not the same as QWidget::isVisible()!
	bool dockVisible;

	// State of the FalseColorings
	QMap<FalseColoring::Type, RgbDockState::Type> coloringState;
	// Calculation progress percentage for progressbar for each FalseColoring
	QMap<FalseColoring::Type, int> coloringProgress;
};

#endif // RGBDOCK_H
