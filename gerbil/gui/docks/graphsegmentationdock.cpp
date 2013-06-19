#include "graphsegmentationdock.h"

GraphSegmentationDock::GraphSegmentationDock(QWidget *parent) :
	QDockWidget(parent)
{
	setupUi(this);
	// TODO initUi
}

GraphSegmentationDock::~GraphSegmentationDock()
{

}

void GraphSegmentationDock::initUi()
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

	connect(graphsegGoButton, SIGNAL(clicked()),
			this, SLOT(startGraphseg()));
	// TODO
//	connect(this, SIGNAL(seedingDone(bool)),
//			graphsegButton, SLOT(setChecked(bool)));
}

// TODO: -> GraphSegModel
void GraphSegmentationDock::runGraphseg(SharedMultiImgPtr input,
							   const vole::GraphSegConfig &config)
{
	/*
	// TODO: why disable GUI? Where is it enabled? -> do it in finishGraphSeg
	// TODO: build signal requestGUI
	setGUIEnabled(false);
	// TODO: should this be a commandrunner instead? arguable..
	BackgroundTaskPtr taskGraphseg(new GraphsegBackground(
		config, input, bandView->seedMap, graphsegResult));
	QObject::connect(taskGraphseg.get(), SIGNAL(finished(bool)),
		this, SLOT(finishGraphSeg(bool)), Qt::QueuedConnection);
	queue.push(taskGraphseg);
	*/
}


// TODO: -> GraphSegModel
void GraphSegmentationDock::finishGraphSeg(bool success)
{
	/*
	 * @ploner probably doesn't work yet (?).
	if (success) {
		// add segmentation to current labeling
		emit alterLabelRequested(bandView->getCurLabel(),
								 *(graphsegResult.get()), false);
		// leave seeding mode for convenience
		emit seedingDone();
	}
	*/
}

// TODO: move this to GraphSegModel (create it)
// multi_img should not be seen by dock/GUI code.
// Only parse GUI parameters into vole::GraphSegConfig here and
// pass it to model via signal requestSegmentation(config).
void GraphSegmentationDock::startGraphseg()
{
	/*
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
	if (src == 0) {
		runGraphseg(image, conf);
	} else if (src == 1) {
		runGraphseg(gradient, conf);
	} else {	// currently shown band, construct from selection in viewport
		representation::t type = viewerContainer->getActiveRepresentation();
		int band = viewerContainer->getSelection(type);
		SharedMultiImgPtr img = viewerContainer->getViewerImage(type);
		SharedDataLock img_lock(img->mutex);
		SharedMultiImgPtr i(new SharedMultiImgBase(
			new multi_img((**img)[band], (*img)->minval, (*img)->maxval)));
		img_lock.unlock();
		runGraphseg(i, conf);
	}
	*/
}
