#ifndef GRAPHSEG_WIDGET_H
#define GRAPHSEG_WIDGET_H

#include "ui_graphsegwidget.h"
#include "model/representation.h"
#include "autohidewidget.h"

#include <graphseg.h>
#include <shared_data.h>

class AutohideView;

class GraphSegWidget : public AutohideWidget, private Ui::GraphSegWidget
{
	Q_OBJECT
	
public:
	// parameter view is needed to set up our custom combo boxes (via initUi)
	explicit GraphSegWidget(AutohideView *view);
	~GraphSegWidget();

protected:
	// parameter view is needed to set up our custom combo boxes
	void initUi(AutohideView *view);

signals:
	void requestShowAndRefreshSeedMap(cv::Mat1s seeding);
	void requestToggleSeedMode(bool toggle);
	void requestClearSeeds();
	void requestLoadSeeds();
	void requestGraphseg(representation::t type,
						 const vole::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegCurBand(const vole::GraphSegConfig &config,
								bool resetLabel);

public slots:
	void processSeedingDone();

protected slots:
	void startGraphseg();
};

#endif // GRAPHSEG_WIDGET_H
