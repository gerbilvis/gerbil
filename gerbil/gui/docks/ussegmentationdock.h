#ifndef USSEGMENTATIONDOCK_H
#define USSEGMENTATIONDOCK_H

#include <QDockWidget>
#include "ui_ussegmentationdock.h"

#include <map>
#include <boost/any.hpp>


class CommandRunner;

/** Unsupervised segmenation dock */

// FIXME SEGMENTATION_TODO
// 2013-06-17 altmann
// Code pulled from MainWindow. Most of this code should go into a model class.
// Not done now because of time constraints.
class UsSegmentationDock : public QDockWidget, private Ui::UsSegmentationDock
{
	Q_OBJECT
public:
	explicit UsSegmentationDock(QWidget *parent = 0);
	
	void segmentationApply(std::map<std::string, boost::any> output);
	void segmentationFinished();
	void startUnsupervisedSeg(bool findKL);
	void startFindKL();
	void unsupervisedSegCancelled();
	void usBandwidthMethodChanged(const QString &current);
	void initUi(size_t nbands);
	void usMethodChanged(int idx);
	void usInitMethodChanged(int idx);
signals:
	
public slots:
private:
	void initUi();

	CommandRunner *usRunner;
};

#endif // USSEGMENTATIONDOCK_H
