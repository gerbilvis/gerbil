#include "model/labelingmodel.h"
#include "iogui.h"

#include <qtopencv.h>

#include "labels/icontask.h"

#define GGDBG_MODULE
#include "../gerbil_gui_debug.h"

LabelingModel::LabelingModel()
	: iconSize(QSize(32,32)), applyROI(true), iconTaskp(NULL)
{
	qRegisterMetaType<QVector<QImage> >("QVector<QImage>");
	//qRegisterMetaType<const QVector<QImage>& >("const QVector<QImage>&");
	labels = full_labels;
}

void LabelingModel::setDimensions(unsigned int height, unsigned int width)
{
	full_labels = cv::Mat1s(height, width, (short)0);
	labels = full_labels;
}

void LabelingModel::updateROI(const cv::Rect &roi)
{
	if (full_labels.empty())
		return;

	labels = cv::Mat1s(full_labels, roi);

	// signal new matrix
	emit newLabeling(labels, colors);

	if (applyROI) {
		invalidateMaskIcons();
	}
}

void LabelingModel::setLabels(const Labeling &labeling, bool full)
{
	// check for implicit full update
	full = full ||
			(labels.cols == full_labels.cols
			 && labels.rows == full_labels.rows);

	// the label matrix
	cv::Mat1s m = labeling();
	if (full) {
		// labeling covers full image
		assert(full_labels.size == m.size);
		m.copyTo(full_labels);
	} else {
		// only current ROI is updated
		assert(labels.size == m.size);
		m.copyTo(labels);
	}

	/* Do not accidentially overwrite full label colors: unintuitive, segfault
	 * Example case: global segmentation performed on ROI; when ROI is changed
	 * afterwards, pixels outside the old ROI still hold their old labels.
	 */
	if (!full && colors.size() >= labeling.colors().size())
	{
		emit newLabeling(labels, colors, false);
		invalidateMaskIcons();
		return;
	}

	/* Alternative case: full label swap or need to overwrite label colors */

	// ensure there is always one foreground color
	if (labeling.colors().size() < 2) {
		// set label colors, but do not emit signal (we need combined signal)
		setLabelColors(Labeling::colors(2, true), false);
	} else {
		// set label colors, but do not emit signal (we need combined signal)
		setLabelColors(labeling.colors(), false);
	}

	// now signal new labels and colors as well
	emit newLabeling(labels, colors, true);
	invalidateMaskIcons();
}

void LabelingModel::setLabels(const cv::Mat1s &labeling)
{
	assert(labeling.size == labels.size);
	// create vole labeling object (build colors) and
	// note: this iterates over the whole label matrix without concurrency.
	// OPTIMIZE
	Labeling vl(labeling, false);
	setLabels(vl, false);
}

void LabelingModel::setLabelColors(const std::vector<cv::Vec3b> &newColors,
								   bool emitSignal)
{
	QVector<QColor> col = Vec2QColor(newColors);
	col[0] = Qt::white; // override black for label 0

	colors.swap(col); // set colors, but also keep old ones around for below

	if (!emitSignal)
		return;

	/* when changed == false, only new colors were added
	 * and most parts of the software do not care about new, unused colors.
	 * 2013-07-22 altmann: Now the LabelDock does. Not changing current behaviour.
	 */
	bool changed = false;
	for (int i = 1; i < colors.size() && i < col.size(); ++i) {
		if (col.at(i) != colors.at(i))
			changed = true;
	}

	// signal only the colors, do not cause costly updates
	emit newLabeling(cv::Mat1s(), colors, changed);
	invalidateMaskIcons();
}

void LabelingModel::addLabel()
{
	// we always have at least one background color
	int labelcount = std::max(colors.count(), 1);

	// increment colors by 1 (add label)
	labelcount++;
	setLabelColors(Labeling::colors(labelcount, true));
}

void LabelingModel::alterLabel(short index, cv::Mat1b mask,
							   bool negative)
{
	if (mask.empty()) {  // clear label
		mask = (labels == index);
		labels.setTo(0, mask);
	} else if (negative) { // remove pixels from label
		mask = mask.mul(labels == index);
		labels.setTo(0, mask);
	} else { // add pixels to label
		labels.setTo(index, mask);
	}

	// signal change
	emit partialLabelUpdate(labels, mask);
	invalidateMaskIcons();
}

void LabelingModel::alterPixels(const cv::Mat1s &newLabels,
								const cv::Mat1b &mask)
{
	// replace pixels
	newLabels.copyTo(labels, mask);

	// signal change
	emit partialLabelUpdate(labels, mask);
	invalidateMaskIcons();
}

void LabelingModel::loadLabeling(const QString &filename)
{
	/* we are properly initialized with the image dimensions
	 * so we take these frome our initialization */
	int height = full_labels.rows;
	int width = full_labels.cols;

	IOGui io("Labeling Image File", "labeling image");
	cv::Mat input = io.readFile(filename, -1, height, width);
	if (input.empty())
		return;

	Labeling labeling(input, false);
	setLabels(labeling, true);
}

void LabelingModel::saveLabeling(const QString &filename)
{
	Labeling labeling(full_labels);
	cv::Mat3b output = labeling.bgr();

	IOGui io("Labeling As Image File", "labeling image");
	io.writeFile(filename, output);
}

void LabelingModel::loadSeeds()
{
	IOGui io("Seed Image File", "seed image");
	// TODO
	/*cv::Mat1s seeding = io.readFile(QString(), 0,
									dimensions.height, dimensions.width);
	if (seeding.empty())
		return;

	bandView->seedMap = seeding;

	// now make sure we are in seed mode
	if (graphsegButton->isChecked()) {
		bandView->refresh();
	} else {
		graphsegButton->toggle();
	}*/
}

void LabelingModel::mergeLabels(const QVector<int> &mlabels)
{
	if(mlabels.size() < 2)
		return;
	QVector<int> xmlabels = mlabels;

	// sort the labels to merge by id
	qSort(xmlabels);

	// target: the label to merge into
	short target = xmlabels[0];

	// mask: all pixels which are to be merged into the target label
	cv::Mat1b mask = cv::Mat1b::zeros(full_labels.rows, full_labels.cols);


	// build mask and new color array
	for(int i=1; i<xmlabels.size(); i++) {
		short label = xmlabels.at(i);
		cv::Mat1b dmask = (full_labels == label);
		mask = mask | dmask;
	}

	full_labels.setTo(target, mask);

	emit newLabeling(labels, colors, false);
	invalidateMaskIcons();
}

void LabelingModel::deleteLabels(const QVector<int> &labels)
{
	// just merge with background, same effect
	QVector<int> tmp = labels;
	tmp.append(0);
	mergeLabels(tmp);
}

void LabelingModel::consolidate()
{
	/* create new labeling that will search for all set labels
	 * it only does this if it gets rgb input
	 * TODO: write a real consolidate function in core/labeling class!
	 */
	Labeling newfull(full_labels);
	newfull.consolidate();
	// get rid of old colors
	//newfull.setColors(Labeling::colors(newfull.colors().size(), true));
	// set it
	setLabels(newfull, true);
}

void LabelingModel::setLabelIconSize(int width, int height)
{
	setLabelIconSize(QSize(width, height));
}

void LabelingModel::setLabelIconSize(const QSize& size)
{
	//GGDBGM("new size " << size << endl);
	iconSize = size;
	invalidateMaskIcons();
}

void LabelingModel::setApplyROI(bool applyROI)
{
	this->applyROI = applyROI;
}

void LabelingModel::computeLabelIcons()
{
	//GGDBG_CALL();
	if (iconTaskp != NULL) {
		if (iconSize != iconTaskp->getIconSize() ||
				applyROI != iconTaskp->getApplyROI())
		{
			//GGDBGM("running IconTask is using old icon size, restarting." << endl);
			discardIconTask();
			startIconTask();
		} else {
			//GGDBGM("IconTask already in progress" << endl);
		}
	} else {
		startIconTask();
	}
}

void LabelingModel::processLabelIconsComputed(const QVector<QImage> &icons)
{
	// The reference argument icons is valid, since the icon task will be
	// deleted only after this function leaves its scope.
	//GGDBGM("IconTask finished successfully" << endl);

	// store the result and emit
	this->icons = icons;
	emit labelIconsComputed(icons);
	discardIconTask();
}

void LabelingModel::processIconTaskAborted()
{
	//GGDBGM("IconTask has aborted" << endl);

	discardIconTask();

	//GGDBGM("requesting new icon computation" << endl);
	computeLabelIcons();
}

void LabelingModel::resetIconTaskPointer()
{
	iconTaskp = NULL;
}

void LabelingModel::invalidateMaskIcons()
{
	//GGDBGM("aborting IconTask" << endl);
	if(iconTaskp) {
		iconTaskp->abort();
	}
	icons.clear();
}

void LabelingModel::startIconTask()
{
	//GGDBGM("starting IconTask." << endl);
	// shared pointer
	IconTaskCtxPtr ctxp(new IconTaskCtx(
				colors.size(),
				full_labels,
				labels,
				iconSize,
				applyROI,
				colors));

	assert(NULL == iconTaskp);
	iconTaskp = new IconTask(ctxp,this);

	connect(iconTaskp, SIGNAL(finished()), this, SLOT(resetIconTaskPointer()));
	connect(iconTaskp, SIGNAL(finished()), iconTaskp, SLOT(deleteLater()));
	connect(iconTaskp, SIGNAL(taskAborted()),
			this, SLOT(processIconTaskAborted()));
	connect(iconTaskp, SIGNAL(labelIconsComputed(const QVector<QImage>&)),
			this, SLOT(processLabelIconsComputed(const QVector<QImage>&)));

	iconTaskp->start();
}


void LabelingModel::discardIconTask()
{
	//GGDBG_CALL();
	if(!iconTaskp)
		return;

	iconTaskp->abort();
	// disconnect all signals, except finished() -> deleteLater()
	disconnect(this, 0, iconTaskp, 0);
	disconnect(iconTaskp, 0, this, 0);
	// old icon task will delete itself after aborting, we can drop the pointer.
	iconTaskp = NULL;
}
