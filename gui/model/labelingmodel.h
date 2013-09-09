#ifndef LABELING_MODEL_H
#define LABELING_MODEL_H

#include <labeling.h>
#include <opencv2/core/core.hpp>
#include <QObject>
#include <QColor>
#include <QVector>

class LabelingModel : public QObject
{
	Q_OBJECT
public:
	explicit LabelingModel();

public slots:
	void setDimensions(unsigned int height, unsigned int width);
	void updateROI(const cv::Rect &roi);
	void setLabels(const vole::Labeling &labeling, bool full);
	void setLabels(const cv::Mat1s &labeling);
	// FIXME should _NOT_ be part of the public API (emitSignal)
	void setLabelColors(const std::vector<cv::Vec3b> &newColors,
						bool emitSignal = true);
	// either clear label, or with other arguments, add/remove pixels to/from it
	void alterLabel(short index, cv::Mat1b mask = cv::Mat1b(),
					bool negative = false);
	// change label of pixels in mask
	void alterPixels(const cv::Mat1s &newLabels, const cv::Mat1b &mask);
	// increment number of available labels
	int addLabel();

	void loadLabeling(const QString &filename = QString());
	void saveLabeling(const QString &filename = QString());
	void loadSeeds();
	/** Merge the labels with indexes from labels.
	 *
	 * @param labels Ids of the labels in the label matix which sould be merged.
	 */
	void mergeLabels(const QVector<int>& labels);
	/** Empty all labels in list (the labels stay, but pixel become background)
	 */
	void deleteLabels(const QVector<int>& labels);

	/** Remove all empty labels, recolor all labels. */
	void consolidate();

signals:
	/** The Labeling has changed.
	 *
	 * If labels == cv::Mat1s(), then only the color vector changed.
	 * @param colorsChanged True only if a color in the previous color vector
	 * changed. If a label is added, colorsChanged is false.
	 */
	void newLabeling(const cv::Mat1s &labels,
					 const QVector<QColor>& colors,
					 bool colorsChanged = false);
	/** new label matrix, but only specific pixels (in mask) changed label
	 */
	void partialLabelUpdate(const cv::Mat1s &labels, const cv::Mat1b &mask);

private:
	// full image labels and roi scoped labels
	/* labels is always a header with the same data as full_labels (CV memory
	 * sharing and reference counting). That is, the contents of labels and
	 * full_labels are updated simultaneously, when changing data in either of
	 * the two objects. Be careful not to reassign labels to an independent
	 * copy (see CV docs).
	 */
	cv::Mat1s full_labels, labels;
	// roi
	cv::Rect roi;
	// label colors
	QVector<QColor> colors;
};

#endif // LABELING_MODEL_H
