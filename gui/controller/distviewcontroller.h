#ifndef DISTVIEWCONTROLLER_H
#define DISTVIEWCONTROLLER_H

#include "controller/controller.h"
#include "dist_view/distviewmodel.h"
#include "dist_view/distviewgui.h"
#include <background_task/background_task_queue.h>

#include <QObject>
#include <QVector>
// TODO
// * check if bands can be removed from MainWindow altogether

class DistViewController : public QObject
{
    Q_OBJECT
    
	struct payload {
		payload(representation::t type) : model(type), gui(type) {}

		DistViewModel model;
		DistViewGUI gui;
	};

public:
	explicit DistViewController(Controller *chief,
								BackgroundTaskQueue *taskQueue);
	void init();

	sets_ptr subImage(representation::t type,
					  const std::vector<cv::Rect> &regions, cv::Rect roi);
	void addImage(representation::t type, sets_ptr temp,
				  const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(representation::t type, SharedMultiImgPtr image, cv::Rect roi);

public slots:
	void setActiveViewer(representation::t type);

	void setGUIEnabled(bool enable, TaskType tt);

	// rebinning in model needs to be displayed by viewport
	void processNewBinning(representation::t type);
	// rebinning in model done, also changed the range
	void processNewBinningRange(representation::t type);

	void finishNormRangeImgChange(bool success);
	void finishNormRangeGradChange(bool success);

	void toggleIgnoreLabels(bool toggle);

	void updateLabels(const cv::Mat1s &labels,
					  const QVector<QColor>& colors = QVector<QColor>(),
					  bool colorsChanged = false);
	void updateLabelsPartially(const cv::Mat1s &labels, const cv::Mat1b &mask);

	void changeBinCount(representation::t type, int bins);

	void propagateBandSelection(int band);

	void drawOverlay(int band, int bin);
	void drawOverlay(const std::vector<std::pair<int, int> > &l, int dim);

	// retrieve highlight mask and delegate alter-label request
	void addHighlightToLabel();
	void remHighlightFromLabel();
	// needed for the functions above
	void setCurrentLabel(int index) { currentLabel = index; }

	void pixelOverlay(int y, int x);

signals:
	void toggleLabeled(bool);
	void toggleUnlabeled(bool);
	void toggleSingleLabel(bool toggle);
	void singleLabelSelected(int);

	void setViewportActive(int);
	void folding();

	void viewportAddSelection();
	void viewportRemSelection();

	void pixelOverlayInvalid();

	void normTargetChanged(bool useCurrent);
	void requestOverlay(const cv::Mat1b &mask);

	// a band was selected and should be shown in spatial view
	void bandSelected(representation::t type, int band);

	// add/delete current highlight to a label
	void alterLabelRequested(short,cv::Mat1b,bool);

	// viewport on-the-fly illuminant-correction (only IMG distview)
	void newIlluminantCurve(QVector<multi_img::Value>);
	void toggleIlluminationShown(bool show);

	// distviewmodel/viewport recognition of applied illuminant (only IMG)
	void newIlluminantApplied(QVector<multi_img::Value>);


protected:
	QMap<representation::t, payload*> map;
	representation::t activeView;
	Controller *chief;

	// needed for add/rem to/from label functionality
	int currentLabel;
};

#endif // DISTVIEWCONTROLLER_H
