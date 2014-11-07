#ifndef CLUSTERINGDOCK_H
#define CLUSTERINGDOCK_H

#include <QDockWidget>
#include "ui_clusteringdock.h"

#include <map>
#include <boost/any.hpp>

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
	
	void segmentationApply(std::map<std::string, boost::any> output);

signals:
	void cancelSegmentationRequested();
	// FIXME: get rid of the Command pointer! Use config object of some sort.
	// FIXME: instead of bool gradient use representation::t
	void segmentationRequested(shell::Command *cmd,
								   bool gradient);
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
