#ifndef CLUSTERING_MODEL_H
#define CLUSTERING_MODEL_H

#include <QObject>
#include <QMetaClassInfo>
#include <map>
#include <boost/any.hpp>

#include <shared_data.h>
#include "representation.h"

class CommandRunner;
namespace vole {
	class Command;
}

/* Model class for unsupervised segmenation.
 *
 * @note: segmentation on gradient not implemented.
 */
class ClusteringModel : public QObject
{
	Q_OBJECT
public:
	explicit ClusteringModel();
	~ClusteringModel();
	void setMultiImage(SharedMultiImgPtr image);

signals:
	void progressChanged(int); // passed on from CommandRunner
	void segmentationCompleted();
	void setLabelsRequested(const cv::Mat1s &labeling);
	void resultKL(int k, int l);
public slots:
	// can numbands and gradient be moved to cmd->config?
	void startSegmentation(
			vole::Command *cmd,
			int numbands,
			bool gradient);
	void cancel();
	void processImageUpdate(representation::t type, SharedMultiImgPtr image);
protected slots:
	void onSegmentationCompleted(std::map<std::string,boost::any> output);
protected:
	/** Cancel and disconnect current CommandRunner and schedule it for deletion.
	 *
	 *  This is a quick HACK. ClusteringModel needs to handle state properly.
	 *  Also Meanshift doesn't seem to handle abort, so the thread will just
	 *  continue to burn CPU cycles.
	 */
	void silenceCommandRunner();

	CommandRunner* cmdr;
	// multi image of current ROI
	SharedMultiImgPtr image;
};

#endif // CLUSTERING_MODEL_H
