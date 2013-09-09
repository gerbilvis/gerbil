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
	void processResultKL(int k, int l);
	void processSegmentationCompleted();
protected slots:
	void startUnsupervisedSeg(bool findKL=false);
	void startFindKL();
	void cancel();
	void usMethodChanged(int idx);
	void usInitMethodChanged(int idx);
	void unsupervisedSegCancelled();
	void usBandwidthMethodChanged(const QString &current);
//	void setBandsSpinBoxTouched(bool touched=true);
protected:
	void initUi();

	int nBandsOld;
};

#endif // CLUSTERINGDOCK_H
