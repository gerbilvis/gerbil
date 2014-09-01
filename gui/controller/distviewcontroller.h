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

struct ReprSubscriptions;

class DistViewController : public QObject
{
    Q_OBJECT
    
	struct payload {
		payload(representation::t type) : model(type), gui(type) {}

		DistViewModel model;
		DistViewGUI gui;
	};

public:
	explicit DistViewController(Controller *ctrl,
								BackgroundTaskQueue *taskQueue,
								ImageModel *im);

	~DistViewController();

	void init();
	/** Initialize subscriptions.
	 *
	 * This needs to be called when all subscription signals/slots have been connected.
	 */
	void initSubscriptions();

	sets_ptr subImage(representation::t type,
					  const std::vector<cv::Rect> &regions, cv::Rect roi);
	void addImage(representation::t type, sets_ptr temp,
				  const std::vector<cv::Rect> &regions, cv::Rect roi);

	// Old, we now listen to imageUpdate from ImageModel
	//void setImage(representation::t type, SharedMultiImgPtr image, cv::Rect roi);

public slots:
	void setActiveViewer(representation::t type);

//	void setGUIEnabled(bool enable, TaskType tt);

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

	void processROIChage(cv::Rect roi);
	void processImageUpdate(representation::t repr,
										SharedMultiImgPtr image,
										bool duplicate);

	void processDistviewNeedsBinning(representation::t repr);

	void processPreROISpawn(cv::Rect const & oldroi,
							cv::Rect const & newroi,
							std::vector<cv::Rect> const & sub,
							std::vector<cv::Rect> const & add,
							bool profitable
							);

	void processPostROISpawn(cv::Rect const & oldroi,
							 cv::Rect const & newroi,
							 std::vector<cv::Rect> const & sub,
							 std::vector<cv::Rect> const & add,
							 bool profitable
							 );


	void processDistviewSubscribeRepresentation(QObject *subscriber, representation::t repr);
	void processDistviewUnsubscribeRepresentation(QObject *subscriber, representation::t repr);

signals:
	void toggleLabeled(bool);
	void toggleUnlabeled(bool);
	void toggleSingleLabel(bool toggle);
	void singleLabelSelected(int);

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

	// SUBSCRIPTION FORWARDING
	void subscribeRepresentation(QObject *subscriber, representation::t repr);
	void unsubscribeRepresentation(QObject *subscriber, representation::t repr);

protected:
	void updateBinning(representation::t repr, SharedMultiImgPtr image);

	QMap<representation::t, payload*> map;
	representation::t activeView;
	Controller *ctrl;
	ImageModel *m_im;

	// needed for add/rem to/from label functionality
	int currentLabel;

	// the current ROI
	cv::Rect m_roi;

	bool m_distviewNeedsBinning[representation::REPSIZE];

	// Subscription state of distiviews.
	// There is no other means for the DistViewController to determine
	// wether or not to do BinSet sub/add in ROI change.
	ReprSubscriptions* m_distviewSubs;

	// Pointer to map of vectors of binsets: BinSets that are recycled
	// during a ROI spawn.
	boost::shared_ptr<QMap<representation::t, sets_ptr> > m_roiSets;
};

#endif // DISTVIEWCONTROLLER_H
