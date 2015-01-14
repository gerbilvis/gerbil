#ifndef CLUSTERINGDOCK_H
#define CLUSTERINGDOCK_H

#include "ui_clusteringdock.h"

#include <model/clustering/clusteringmethod.h>
#include <model/clustering/clusteringrequest.h>
#include <model/representation.h>

#include <QDockWidget>

namespace shell {
	class Command;
}
class CommandRunner;

/** Unsupervised segmenation dock */

class ClusteringDock : public QDockWidget, private Ui::ClusteringDock
{
	Q_OBJECT
public:
	explicit ClusteringDock(QWidget *parent = 0);
	
signals:
	void cancelSegmentationRequested();
	void segmentationRequested(const ClusteringRequest &r);
public slots:
	void updateProgress(int percent);
	void processSegmentationCompleted();

protected slots:
	void startUnsupervisedSeg();
	void cancel();
	void usMethodChanged(int idx);
	void unsupervisedSegCancelled();

protected:
	void initUi();
};

#endif // CLUSTERINGDOCK_H
