#include "model/labelingmodel.h"
#include <app/gerbilio.h>
#include <qtopencv.h>

#include "labels/icontask.h"
#include <QSettings>
#include <QDebug>

#include <boost/make_shared.hpp>
#include <algorithm>

#define GGDBG_MODULE
#include "../gerbil_gui_debug.h"

LabelingModel::LabelingModel(QObject *parent)
	: QObject(parent), applyROI(true), iconTask(NULL)
{
	qRegisterMetaType<QVector<QImage> >("QVector<QImage>");

	restoreState();
}

void LabelingModel::restoreState()
{
	QSettings settings;
	applyROI = settings.value("Labeling/applyROI", true).toBool();
}

void LabelingModel::setImageSize(unsigned int height, unsigned int width)
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
		computeLabelIcons();
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
	if (!full && colors.size() >= (int)labeling.colors().size())
	{
		emit newLabeling(labels, colors, false);
		computeLabelIcons();
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
	computeLabelIcons();
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
	computeLabelIcons();
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
	computeLabelIcons();
}

void LabelingModel::alterPixels(const cv::Mat1s &newLabels,
								const cv::Mat1b &mask)
{
	// replace pixels
	newLabels.copyTo(labels, mask);

	// signal change
	emit partialLabelUpdate(labels, mask);
	computeLabelIcons();
}

void LabelingModel::loadLabeling(const QString &filename)
{
	GerbilIO io(nullptr, "Labeling From Image File", "labeling image");
	io.setFileCategory("LabelFile");
	io.setFileSuffix(".png");
	/* we are properly initialized with the image dimensions
	 * so we take these frome our initialization */
	io.setWidth(full_labels.cols);
	io.setHeight(full_labels.rows);
	io.setFileName(filename);
	cv::Mat input = io.readImage();
	if (input.empty())
		return;

	Labeling labeling(input, false);
	setLabels(labeling, true);
}

void LabelingModel::saveLabeling(const QString &filename)
{
	Labeling labeling(full_labels);
	cv::Mat3b output = labeling.bgr();

	GerbilIO io(nullptr, "Labeling As Image File", "labeling image");
	io.setFileCategory("LabelFile");
	io.setFileSuffix(".png");
	io.writeImage(output);
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
	computeLabelIcons();
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

void LabelingModel::setApplyROI(bool applyROI)
{
	this->applyROI = applyROI;
}

void LabelingModel::computeLabelIcons(QSize size)
{
	if (size.isValid()) { // computation due to size change
		if (iconSize == size)
			return; // nothing to do

		iconSize = size;
	}

	if (iconSize.isEmpty()) {
		return; // GUI not fully initialized yet
	}

	//GGDBG_CALL();
	if (iconTask != NULL) {
		iconTask->abort();
		// task will delete itself after aborting, we can drop the pointer.
	}

	//GGDBGM("starting IconTask." << endl);
	// shared pointer
	auto ctx = boost::make_shared<IconTaskCtx>(colors.size(), full_labels,
	                                           labels, iconSize, applyROI,
	                                           colors);
	iconTask = new IconTask(ctx, this);

	connect(iconTask, SIGNAL(finished()), iconTask, SLOT(deleteLater()));
	connect(iconTask, SIGNAL(labelIconsComputed(QVector<QImage>)),
			this, SLOT(processLabelIconsComputed(QVector<QImage>)));

	iconTask->start();
}

void LabelingModel::processLabelIconsComputed(QVector<QImage> icons)
{
	// remove our reference to the task who will delete itself
	iconTask = NULL;

	emit labelIconsComputed(icons); // copies the icons
}

