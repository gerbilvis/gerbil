#ifndef CLUSTERINGDOCK_H
#define CLUSTERINGDOCK_H

#include <QDockWidget>
#include "ui_clusteringdock.h"

#include <map>
#include <boost/any.hpp>

namespace vole {
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
	void segmentationRequested(vole::Command *cmd,
								   int numbands,
								   bool gradient);
public slots:
	void setNumBands(int nBands);
	void updateProgress(int percent);
	void processSegmentationCompleted();

protected slots:
	void startUnsupervisedSeg();
	void cancel();
	void usMethodChanged(int idx);
	void unsupervisedSegCancelled();

protected:
	void initUi();

	int nBandsOld;
};

#endif // CLUSTERINGDOCK_H
