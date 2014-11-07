#ifndef CLUSTERING_MODEL_H
#define CLUSTERING_MODEL_H

#include <QObject>
#include <QMetaClassInfo>
#include <map>
#include <boost/any.hpp>

#include <shared_data.h>
#include "representation.h"

namespace shell {
	class Command;
}
class CommandRunner;

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

signals:

	void progressChanged(int); // passed on from CommandRunner
	void segmentationCompleted();
	void setLabelsRequested(const cv::Mat1s &labeling);
	void resultKL(int k, int l);

	void subscribeRepresentation(QObject *subscriber, representation::t repr);
	void unsubscribeRepresentation(QObject *subscriber, representation::t repr);

public slots:

	// can numbands and gradient be moved to cmd->config?
	void requestSegmentation(shell::Command *cmd,
			bool gradient);
	void cancel();

	void processImageUpdate(representation::t repr,
							SharedMultiImgPtr image,
							bool duplicate);

protected slots:

	void processSegmentationCompleted(std::map<std::string,boost::any> output);
	void processSegmentationFailed();

protected:

	/** Reset to initial (idle) state and unsubscribe representations. */
	void resetToIdle();

	/** Cancel and disconnect current CommandRunner and schedule it for deletion.
	 *
	 *  This is a quick HACK. ClusteringModel needs to handle state properly.
	 *  Also Meanshift doesn't seem to handle abort, so the thread will just
	 *  continue to burn CPU cycles.
	 */
	void abortCommandRunner();

	// Actually kick-off the computation.
	void startSegmentation();

	struct State {
		enum t { Idle, Subscribed, Executing };
	};

	State::t state;

	// FIXME This is a workaround to store requestSegmentation args.
    // Ideally requestSegmentation() should be passed _one_ configuration object,
    // which is not a pointer (or at least a smart pointer), to be stored here.
	struct Request {
		shell::Command *cmd;
		bool gradient;
		representation::t repr;
	};

	// Pending request.
	boost::shared_ptr<Request> request;

	CommandRunner* commandRunner;

	// IMG and GRAD representation of multi image of current ROI.
	QMap<representation::t, SharedMultiImgPtr> inputMap;
};

#endif // CLUSTERING_MODEL_H
