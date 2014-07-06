#ifndef FALSECOLORDOCK_H
#define FALSECOLORDOCK_H

#include "ui_falsecolordock_sel.h"
#include <QDockWidget>
#include <shared_data.h>
#include "../model/representation.h"
#include "../model/falsecolor/falsecoloring.h"

class AutohideView;
class ScaledView;
class AutohideWidget;

struct FalseColoringState {
	enum Type {UNKNOWN=0,FINISHED, CALCULATING, ABORTING};
};
std::ostream &operator<<(std::ostream& os, const FalseColoringState::Type& state);

class FalseColorDock : public QDockWidget {
	Q_OBJECT
public:
	explicit FalseColorDock(QWidget *parent = 0);
	
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
	/** Request re-calculation of non-determinitstic representation (e.g. SOM). */
	void falseColoringRecalcRequested(FalseColoring::Type coloringType);

	void subscribeFalseColoring(QObject* subscriber, FalseColoring::Type coloringType);
	void unsubscribeFalseColoring(QObject* subscriber, FalseColoring::Type coloringType);

public slots:
	void processVisibilityChanged(bool visible);
	void processCalculationProgressChanged(FalseColoring::Type coloringType, int percent);
	void processColoringComputed(FalseColoring::Type coloringType, QPixmap p);
	void processComputationCancelled(FalseColoring::Type coloringType);

	/** Inform the FalseColorDock an update of the FalseColoring coloringType is pending.
	 *
	 * This triggers the display of a progress bar.
	 */
	void setCalculationInProgress(FalseColoring::Type coloringType);
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

	void enterState(FalseColoring::Type coloringType, FalseColoringState::Type state);

	// State of the FalseColorings
	QMap<FalseColoring::Type, FalseColoringState::Type> coloringState;
	// Calculation progress percentage for progressbar for each FalseColoring
	QMap<FalseColoring::Type, int> coloringProgress;
	QMap<FalseColoring::Type, bool> coloringUpToDate;
	FalseColoring::Type lastShown;

	// viewport and scene
	AutohideView *view;
	ScaledView *scene;
	// UI and widget for our button row
	Ui::FalsecolorDockSelUI *uisel;
	AutohideWidget *sel;

};

#endif // FALSECOLORDOCK_H
