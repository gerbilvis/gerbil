#ifndef NORMDOCK_H
#define NORMDOCK_H


#include <QDockWidget>
#include <QMap>

#include "ui_normdock.h"

#include "model/representation.h"

// FIXME: need to include multi_img_tasks.h just for MultiImg::NormMode.
// This is bad dependency mangagement: multi_img_tasks.h is huge.
#include <multi_img_tasks.h>
#include <boost/bind/bind.hpp>


class NormDock : public QDockWidget, protected Ui::NormDock
{
	Q_OBJECT
	
public:
	explicit NormDock(QWidget *parent = 0);
	~NormDock();
	
	void setLimitedMode(bool limited);
public slots:
	void setGuiEnabled(bool enable, TaskType tt);

	// FIXME need a corresponding signal from ImageModel
	void setNormRange(representation::t type, const ImageDataRange& range);

	void setNormMode(representation::t type, MultiImg::NormMode mode);
	void setNormTarget(representation::t type);

protected slots:
	void processApplyClicked();
	void processNormTargetSelected();
	void processNormModeSelected(int idx)
		{ modes[normTarget] = static_cast<MultiImg::NormMode>(idx); }
	void processMinValueChanged();
	void processMaxValueChanged();
signals:
	void normalizationParametersChanged(
			representation::t type,
			MultiImg::NormMode normMode,
			ImageDataRange targetRange
			);
	void applyNormalizationRequested();
	void applyClampDataRequested();

protected:
	// 2013-07-3 altmann: don't know how to handle limitedMode right now,
	//   so I add a member for state here. Maybe we can find a better solution.
	bool limitedMode;

	// store current range for each representation
	QMap<representation::t, ImageDataRange> ranges;

	// store currently selected normaliztation mode for each representation
	QMap<representation::t, MultiImg::NormMode> modes;

	// user edits norm parameters for IMG or GRAD?
	representation::t normTarget;

	void initUi();
};

#endif // NORMDOCK_H
