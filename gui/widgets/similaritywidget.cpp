#include "similaritywidget.h"
#include "sm_config.h"

using IM = ScaledView::InputMode;

SimilarityWidget::SimilarityWidget(AutohideView *view) :
	AutohideWidget()
{
	setupUi(this);
	initUi(view);
}

SimilarityWidget::~SimilarityWidget()
{
}

void SimilarityWidget::initUi(AutohideView *view)
{
	using namespace similarity_measures;

	pickButton->setAction(actionTarget);
	actionTarget->setData(QVariant::fromValue(ScaledView::InputMode::Target));

	connect(actionTarget, SIGNAL(triggered(bool)),
	        actionTarget, SLOT(setEnabled(bool)));

	doneButton->setAction(actionDone);
	actionDone->setData(QVariant::fromValue(ScaledView::InputMode::Zoom));

	similarityBox->setAHView(view);
	similarityBox->addItem("Manhattan distance (L1)", MANHATTAN);
	similarityBox->addItem("Euclidean distance (L2)", EUCLIDEAN);
	similarityBox->addItem(QString::fromUtf8("Chebyshev distance (L∞)"),
	                       CHEBYSHEV);
	similarityBox->addItem("Spectral Angle", SPECTRAL_ANGLE);
	similarityBox->addItem("Spectral Information Divergence",
	                       SPEC_INF_DIV);
	similarityBox->addItem("SID+SAM I", SIDSAM1);
	similarityBox->addItem("SID+SAM II", SIDSAM2);
	similarityBox->addItem("Normalized L2", NORM_L2);
	similarityBox->setCurrentIndex(3);
}

similarity_measures::SMConfig SimilarityWidget::config()
{
	similarity_measures::SMConfig conf;
	conf.function = (similarity_measures::measure)
	                similarityBox->itemData(similarityBox->currentIndex()).value<int>();

	return conf;
}
