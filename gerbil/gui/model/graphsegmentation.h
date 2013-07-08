#ifndef GRAPHSEGMENTATIONMODEL_H
#define GRAPHSEGMENTATIONMODEL_H

#include "representation.h"

#include <shared_data.h>
#include <background_task_queue.h>
#include <opencv2/core/core.hpp>

#include <QObject>
#include <QPixmap>

namespace vole
{
	class GraphSegConfig;
}

class GraphSegmentationModel : public QObject
{
	Q_OBJECT

public:
	GraphSegmentationModel(QObject *parent, BackgroundTaskQueue *queue);
	~GraphSegmentationModel();

	// always set img and grad before using the class
	void setMultiImage(representation::t type, SharedMultiImgPtr image);
	void setSeedMap(cv::Mat1s *seedMap);

protected:
	void startGraphseg(SharedMultiImgPtr input,
					   const vole::GraphSegConfig &config);

public slots:
	void setCurLabel(int curLabel);
//	void setCurBand(representation::t type, int bandId);
	void setCurBand(QPixmap band, QString description);
	void runGraphseg(representation::t type,
					 const vole::GraphSegConfig &config);
	void runGraphsegBand(const vole::GraphSegConfig &config);

protected slots:
	void finishGraphSeg(bool success);

signals:
	/* effect: gerbil GUI enabled/disabled. */
	void setGUIEnabledRequested(bool enable, TaskType tt);

	void alterLabelRequested(short index, const cv::Mat1b &mask, bool negative);
	void seedingDone();

protected:
	typedef QMap<representation::t, SharedMultiImgPtr> ImageMap;

	BackgroundTaskQueue *const queue;
	ImageMap map;
	cv::Mat1s *seedMap;

//	representation::t curRepr;
//	int curBand;
	QPixmap curBand;
	int curLabel;

	boost::shared_ptr<cv::Mat1s> graphsegResult;
};

#endif // GRAPHSEGMENTATIONMODEL_H
