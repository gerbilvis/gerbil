#ifndef ILLUMINATION_MODEL_H
#define ILLUMINATION_MODEL_H

#include <QObject>

#include <boost/shared_ptr.hpp>
#include <vector>

#include <background_task/background_task_queue.h>
#include <multi_img.h>
#include <shared_data.h>
// for representation
#include <model/imagemodel.h>

#include "multi_img/illuminant.h"


class IllumModel : public QObject
{
	Q_OBJECT
public:
	explicit IllumModel(BackgroundTaskQueue *queue);

	void setMultiImage(SharedMultiImgPtr image);
	void setRoi(cv::Rect roi);

signals:
	/* effect: gerbil GUI enabled/disabled. */
	void setGUIEnabledRequested(bool enable, TaskType tt);
	/* effect: rebuild ROI from input full image. */
	void requestInvalidateROI(cv::Rect roi);
	/* effect: illuminant curve is drawn in viewers */
	void newIlluminantCurve(QVector<multi_img::Value> illum);
	/* effect: illuminant is employed in binning */
	void newIlluminantApplied(QVector<multi_img::Value> illum);

public slots:
	void applyIllum();
	void updateIllum1(int t);
	void updateIllum2(int t);

protected slots:
	void finishTask(bool success);

protected:
	// FIXME altmann: reference to member data... asking for trouble
	const Illuminant & getIlluminant(int t);
	QVector<multi_img_base::Value> getIllumCoeff(int t);
	void buildIllum(int t);
	void submitRemoveOldIllumTask();
	void submitAddNewIllumTask();
private:
	// pointer to BackgroundTaskQueue
	BackgroundTaskQueue* queue;

	// pointer to shared pointer to multi_img
	SharedMultiImgPtr image;

	// current region of interest
	cv::Rect roi;

	// cache for illumination coefficients
	typedef std::map<int, std::pair<
			Illuminant, QVector<multi_img::Value> > > Illum_map;
	Illum_map illuminants;

	// Selected illuminant temp (K) in the combo boxes
	int i1, i2;
};

#endif // MODEL_ILLUMINATION

