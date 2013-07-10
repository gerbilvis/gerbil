#ifndef GRAPHSEGMENTATIONDOCK_H
#define GRAPHSEGMENTATIONDOCK_H

#include "ui_graphsegwidget.h"
#include "../model/representation.h"

#include <graphseg.h>
#include <shared_data.h>

#include <QDockWidget>

class GraphSegWidget : public QWidget, private Ui::GraphSegWidget
{
	Q_OBJECT
	
public:
	explicit GraphSegWidget(QWidget *parent = 0);
	~GraphSegWidget();

protected:
	void initUi();

signals:
	void requestGraphseg(representation::t, const vole::GraphSegConfig &config,
						 bool resetLabel);
	void requestGraphsegCurBand(const vole::GraphSegConfig &config,
								bool resetLabel);

protected slots:
	void startGraphseg();
};

#endif // GRAPHSEGMENTATIONDOCK_H
