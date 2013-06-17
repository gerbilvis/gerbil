#ifndef GRAPHSEGMENTATIONDOCK_H
#define GRAPHSEGMENTATIONDOCK_H

#include <QDockWidget>

#include <graphseg.h>

#include "../../common/shared_data.h"

#include "ui_graphsegmentationdock.h"

class GraphSegmentationDock : public QDockWidget, private Ui::GraphSegmentationDock
{
	Q_OBJECT
	
public:
	explicit GraphSegmentationDock(QWidget *parent = 0);
	~GraphSegmentationDock();
	

protected:
	void initUi();
	void startGraphseg();
	void finishGraphSeg(bool success);
	void runGraphseg(SharedMultiImgPtr input, const vole::GraphSegConfig &config);
private:

};

#endif // GRAPHSEGMENTATIONDOCK_H
