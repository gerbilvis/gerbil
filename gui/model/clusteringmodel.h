#ifndef CLUSTERING_MODEL_H
#define CLUSTERING_MODEL_H

#include <QObject>
#include <QMetaClassInfo>
#include <map>
#include <boost/any.hpp>

#include <shared_data.h>

#include <model/representation.h>
#include <model/clustering/clusteringmethod.h>
#include <model/clustering/clusteringrequest.h>


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
	/** Request segmentation with ClusteringMethod method on
	 * representation repr.
	 *
	 * Only NORM and GRAD are supported as representations.
	 */
	void requestSegmentation(const ClusteringRequest &r);
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

	/** Cancel and disconnect current CommandRunner and schedule it for
	 *  deletion. */
	void abortCommandRunner();

	// Actually kick-off the computation.
	void startSegmentation();

	struct State {
		enum t { Idle, Subscribed, Executing };
	};

	State::t state;

	// Pending request.
	boost::shared_ptr<ClusteringRequest> request;

	// The CommandRunner when executing.
	CommandRunner* commandRunner;

	// IMG and GRAD representation of multi image of current ROI.
	QMap<representation::t, SharedMultiImgPtr> inputMap;
};

#endif // CLUSTERING_MODEL_H
