#include "widgets/graphsegwidget.h"
#include "model/representation.h"

GraphSegWidget::GraphSegWidget(AutohideView *view) :
	AutohideWidget()
{
	setupUi(this);
	initUi(view);
}

GraphSegWidget::~GraphSegWidget() { }

void GraphSegWidget::initUi(AutohideView *view)
{
	sourceBox->setAHView(view);
	sourceBox->addItem("Image", 0);
	sourceBox->addItem("Gradient", 1); // TODO PCA
	sourceBox->addItem("Shown Band", 2);
	sourceBox->setCurrentIndex(0);

	similarityBox->setAHView(view);
	similarityBox->addItem("Manhattan distance (L1)", vole::MANHATTAN);
	similarityBox->addItem("Euclidean distance (L2)", vole::EUCLIDEAN);
	similarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"),
								   vole::CHEBYSHEV);
	similarityBox->addItem("Spectral Angle", vole::MOD_SPEC_ANGLE);
	similarityBox->addItem("Spectral Information Divergence",
								   vole::SPEC_INF_DIV);
	similarityBox->addItem("SID+SAM I", vole::SIDSAM1);
	similarityBox->addItem("SID+SAM II", vole::SIDSAM2);
	similarityBox->addItem("Normalized L2", vole::NORM_L2);
	similarityBox->setCurrentIndex(3);

	seedModeWidget->setHidden(true);

	connect(seedModeButton, SIGNAL(toggled(bool)),
			this, SIGNAL(requestToggleSeedMode(bool)));

	// first hide button, then show seed explanations
	connect(seedModeButton, SIGNAL(toggled(bool)),
			loadSeedsButton, SLOT(setHidden(bool)));
	connect(seedModeButton, SIGNAL(toggled(bool)),
			seedModeWidget, SLOT(setShown(bool)));

	connect(clearSeedsButton, SIGNAL(clicked()),
			this, SIGNAL(requestClearSeeds()));
	connect(loadSeedsButton, SIGNAL(clicked()),
			this, SIGNAL(requestLoadSeeds()));

	connect(addButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
	connect(replaceButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
}

// pass it to model via signal requestSegmentation(config).
void GraphSegWidget::startGraphseg()
{
	bool resetLabel = (sender() == replaceButton);

	vole::GraphSegConfig conf("graphseg");
	conf.algo = vole::PRIM;
	conf.similarity.measure = (vole::similarity_fun)
		  similarityBox->itemData(similarityBox->currentIndex()).value<int>();
#ifdef WITH_EDGE_DETECT
	conf.som_similarity = false;
#endif
	conf.geodesic = false;
	conf.multi_seed = false;
	int src = sourceBox->itemData(sourceBox->currentIndex()).value<int>();

	if (src == 0) {
		emit requestGraphseg(representation::IMG, conf, resetLabel);
	} else if (src == 1) {
		emit requestGraphseg(representation::GRAD, conf, resetLabel);
	} else {
		emit requestGraphsegCurBand(conf, resetLabel); // currently shown band
	}
}

void GraphSegWidget::processSeedingDone()
{
	seedModeButton->setChecked(false);
}
