#ifndef NORMDOCK_H
#define NORMDOCK_H


#include <QDockWidget>
#include <QMap>

#include "ui_normdock.h"

#include "model/representation.h"

// FIXME: need to include background_task/tasks/multi_img_tasks.h just for MultiImg::NormMode.
// This is bad dependency mangagement: background_task/tasks/multi_img_tasks.h is huge.
#include <background_task/tasks/multi_img_tasks.h>


// FIXME normalization functionality
//
// * range is initialized incorrectly from ImageModel::dataRangeUdpate().
//   This appears to be a bug in CommonTbb::DetermineRange ?
//
// * What is the purpose and desired semantics of the different normalization
//   modes? How should the GUI behave?
//
// * data clamping is not implemented (NormDock and ImageModel).

class NormDock : public QDockWidget, protected Ui::NormDock
{
	Q_OBJECT
	
public:
	explicit NormDock(QWidget *parent = 0);
	~NormDock();
	
	void setLimitedMode(bool limited);
public slots:
	// TODO connect
	void setGuiEnabled(bool enable, TaskType tt);

	void setNormRange(representation::t type, const multi_img::Range& range);

	// setNormMode, setNormTarget unused and untested.
	void setNormMode(representation::t type, MultiImg::NormMode mode);
	void setNormTarget(representation::t type);

protected slots:
	void processApplyClicked();
	/** Update GUI elements from data, except normTarget buttons. */
	void updateGUI();
	void processNormModeSelected(int idx);
	void processMinValueChanged();
	void processMaxValueChanged();
signals:
	// Request actual data ranges from model, see setNormRange()
	void computeDataRangeRequested(representation::t type);
	void normalizationParametersChanged(
			representation::t type,
			MultiImg::NormMode normMode,
			multi_img::Range targetRange
			);
	void applyNormalizationRequested();
	void applyClampDataRequested();

protected:
	// 2013-07-3 altmann: don't know how to handle limitedMode right now,
	//   so I add a member for state here. Maybe we can find a better solution.
	bool limitedMode;

	// store current range for each representation
	QMap<representation::t, multi_img::Range> ranges;

	// store currently selected normaliztation mode for each representation
	QMap<representation::t, MultiImg::NormMode> modes;

	// user edits norm parameters for IMG or GRAD?
	representation::t normTarget;

	void initUi();
};

#endif // NORMDOCK_H
