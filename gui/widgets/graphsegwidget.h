#ifndef GRAPHSEG_WIDGET_H
#define GRAPHSEG_WIDGET_H

#include "ui_graphsegwidget.h"
#include "model/representation.h"
#include "autohidewidget.h"

#include <graphseg.h>
#include <shared_data.h>

class GraphSegWidget : public AutohideWidget, private Ui::GraphSegWidget
{
	Q_OBJECT
	
public:
	explicit GraphSegWidget(QWidget *parent = 0);
	~GraphSegWidget();

protected:
	void initUi();

signals:
	void requestShowAndRefreshSeedMap(cv::Mat1s seeding);
	void requestLoadSeeds();
	void requestGraphseg(representation::t type,
						 const vole::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegCurBand(const vole::GraphSegConfig &config,
								bool resetLabel);

protected slots:
	void startGraphseg();
};

#endif // GRAPHSEG_WIDGET_H
