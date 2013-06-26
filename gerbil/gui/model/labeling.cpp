#include "model/labeling.h"
#include "iogui.h"

#include <qtopencv.h>

#include "../gerbil_gui_debug.h"

LabelingModel::LabelingModel()
{
}

void LabelingModel::setDimensions(unsigned int height, unsigned int width)
{
	full_labels = cv::Mat1s(height, width, (short)0);
}

void LabelingModel::updateROI(const cv::Rect &roi)
{
	if (full_labels.empty())
		return;

	labels = cv::Mat1s(full_labels, roi);

	// signal new matrix
	emit newLabeling(labels);
}

void LabelingModel::setLabels(const vole::Labeling &labeling, bool full)
{
	if (full) {
		// labeling covers full image
		full_labels = labeling();
	} else {
		// only current ROI is updated
		labels = labeling();
	}

	// set label colors, but do not emit signal (we need combined signal)
	setLabelColors(labeling.colors(), false);

	// now signal new labels and colors as well
	emit newLabeling(labels, colors, true);
}

void LabelingModel::setLabels(const cv::Mat1s &labeling)
{
	assert(labeling.size == labels.size);
	labels = labeling.clone();
	emit newLabeling(labels);
}

void LabelingModel::setLabelColors(const std::vector<cv::Vec3b> &newColors,
								   bool emitSignal)
{
	QVector<QColor> col = vole::Vec2QColor(newColors);
	col[0] = Qt::white; // override black for label 0

	colors.swap(col); // set colors, but also keep old ones around for below

	if (!emitSignal)
		return;

	/* when changed == false, only new colors were added
	 * and most parts of the software do not care about new, unused colors */
	bool changed = false;
	for (int i = 1; i < colors.size() && i < col.size(); ++i) {
		if (col.at(i) != colors.at(i))
			changed = true;
	}

	// signal only the colors, do not cause costly updates
	emit newLabeling(cv::Mat1s(), colors, changed);
}

int LabelingModel::addLabel()
{
	//GGDBG_CALL();
	// we always have at least one background color
	int index = std::max(colors.count(), 1);

	// increment colors by 1
	setLabelColors(vole::Labeling::colors(index + 1, true));

	// return index of new label (count - 1)
	return index;
}

void LabelingModel::alterLabel(short index, cv::Mat1b mask,
							   bool negative)
{
	int operation = 1; // add pixels to label
	if (mask.empty()) {
		operation = 0; // clear label
		mask = (labels == index);
	} else if (negative) {
		operation = -1; // remove pixels from label
	}

	switch (operation) {
	case 0:
		// clear the label
		labels.setTo(0, mask);
		break;
	case 1:
		// add pixels to label
		labels.setTo(index, mask);
		break;
	case -1:
		// remove pixels from label (not from other labels)
		mask = mask.mul(labels == index);
		labels.setTo(0, mask);
		break;
	default: while(0); // no compiler complaining
	}

	// signal change
	emit partialLabelUpdate(labels, mask);
}

void LabelingModel::alterPixels(const cv::Mat1s &newLabels,
								const cv::Mat1b &mask)
{
	// replace pixels
	newLabels.copyTo(labels, mask);

	// signal change
	emit partialLabelUpdate(labels, mask);
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

	vole::Labeling labeling(input, false);
	setLabels(labeling, true);
}

void LabelingModel::saveLabeling(const QString &filename)
{
	vole::Labeling labeling(full_labels);
	cv::Mat3b output = labeling.bgr();

	IOGui io("Labeling As Image File", "labeling image");
	io.writeFile(filename, output);
}
