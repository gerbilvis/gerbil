#ifndef USSEGMENTATIONDOCK_H
#define USSEGMENTATIONDOCK_H

#include <QDockWidget>
#include "ui_ussegmentationdock.h"

#include <map>
#include <boost/any.hpp>

namespace vole {
	class Command;
}
class CommandRunner;

/** Unsupervised segmenation dock */

class UsSegmentationDock : public QDockWidget, private Ui::UsSegmentationDock
{
	Q_OBJECT
public:
	explicit UsSegmentationDock(QWidget *parent = 0);
	
	void segmentationApply(std::map<std::string, boost::any> output);
	void initUi(size_t nbands);

signals:
	void cancelSegmentationRequested();
	void segmentationRequested(vole::Command *cmd,
								   int numbands,
								   bool gradient);
public slots:
	int updateProgress(int percent);
	int processResultKL(int k, int l);
	void processSegmentationCompleted();
protected slots:
	void startUnsupervisedSeg(bool findKL=false);
	void startFindKL();
	void cancel();
	void usMethodChanged(int idx);
	void usInitMethodChanged(int idx);
	void unsupervisedSegCancelled();
	void usBandwidthMethodChanged(const QString &current);
private:
	void initUi();
};

#endif // USSEGMENTATIONDOCK_H
