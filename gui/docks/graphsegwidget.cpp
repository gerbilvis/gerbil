#include "graphsegwidget.h"
#include "../model/representation.h"

GraphSegWidget::GraphSegWidget(QWidget *parent) :
	QWidget(parent)
{
	setupUi(this);
	initUi();
}

GraphSegWidget::~GraphSegWidget() { }

void GraphSegWidget::initUi()
{
	graphsegSourceBox->addItem("Image", 0);
	graphsegSourceBox->addItem("Gradient", 1); // TODO PCA
	graphsegSourceBox->addItem("Shown Band", 2);
	graphsegSourceBox->setCurrentIndex(0);

	graphsegSimilarityBox->addItem("Manhattan distance (L1)", vole::MANHATTAN);
	graphsegSimilarityBox->addItem("Euclidean distance (L2)", vole::EUCLIDEAN);
	graphsegSimilarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"),
								   vole::CHEBYSHEV);
	graphsegSimilarityBox->addItem("Spectral Angle", vole::MOD_SPEC_ANGLE);
	graphsegSimilarityBox->addItem("Spectral Information Divergence",
								   vole::SPEC_INF_DIV);
	graphsegSimilarityBox->addItem("SID+SAM I", vole::SIDSAM1);
	graphsegSimilarityBox->addItem("SID+SAM II", vole::SIDSAM2);
	graphsegSimilarityBox->addItem("Normalized L2", vole::NORM_L2);
	graphsegSimilarityBox->setCurrentIndex(3);

	graphsegAlgoBox->addItem("Kruskal", vole::KRUSKAL);
	graphsegAlgoBox->addItem("Prim", vole::PRIM);
	graphsegAlgoBox->addItem("Power Watershed q=2", vole::WATERSHED2);
	graphsegAlgoBox->setCurrentIndex(1);

	connect(loadSeedsButton, SIGNAL(clicked()),
			this, SIGNAL(requestLoadSeeds()));

	connect(graphsegGoButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
}

// pass it to model via signal requestSegmentation(config).
void GraphSegWidget::startGraphseg()
{
	vole::GraphSegConfig conf("graphseg");
	conf.algo = (vole::graphsegalg)
				graphsegAlgoBox->itemData(graphsegAlgoBox->currentIndex())
				.value<int>();
	conf.similarity.measure = (vole::similarity_fun)
		  graphsegSimilarityBox->itemData(graphsegSimilarityBox->currentIndex())
		  .value<int>();
#ifdef WITH_EDGE_DETECT
	conf.som_similarity = false;
#endif
	conf.geodesic = graphsegGeodCheck->isChecked();
	conf.multi_seed = false;
	int src = graphsegSourceBox->itemData(graphsegSourceBox->currentIndex())
								 .value<int>();

	bool resetLabel = resetLabelRadio->isChecked();

	if (src == 0) {
		emit requestGraphseg(representation::IMG, conf, resetLabel);
	} else if (src == 1) {
		emit requestGraphseg(representation::GRAD, conf, resetLabel);
	} else {
		emit requestGraphsegCurBand(conf, resetLabel); // currently shown band
	}
}
