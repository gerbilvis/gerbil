#ifndef MODEL_LABELING_H
#define MODEL_LABELING_H

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
	
	void loadLabeling(const QString &filename = QString());
	void saveLabeling(const QString &filename = QString());

public slots:
	void setDimensions(unsigned int height, unsigned int width);
	void updateROI(const cv::Rect &roi);
	void setLabels(const vole::Labeling &labeling, bool full);
	void setLabels(const cv::Mat1s &labeling);
	void setLabelColors(const std::vector<cv::Vec3b> &newColors,
						bool emitSignal = true);
	// either clear label, or with other arguments, add/remove pixels to/from it
	void alterLabel(short index, cv::Mat1b mask = cv::Mat1b(),
					bool negative = false);
	// change label of pixels in mask
	void alterPixels(const cv::Mat1s &newLabels, const cv::Mat1b &mask);
	// increment number of available labels
	int addLabel();
signals:
	/** entirely new label matrix, or different label colors, or both
	 * @arg changedColors true if previously existing colors changed
	 */
	void newLabeling(const cv::Mat1s &labels,
					 const QVector<QColor>& colors = QVector<QColor>(),
					 bool colorsChanged = false);
	/** new label matrix, but only specific pixels (in mask) changed label
	 */
	void partialLabelUpdate(const cv::Mat1s &labels, const cv::Mat1b &mask);

private:
	// full image labels and roi scoped labels
	/* note that due to CV memory sharing, all other parties autonomously update
	 * the content of labels (and full_labels simultaneously)
	 */
	cv::Mat1s full_labels, labels;
	// roi
	cv::Rect roi;
	// label colors
	QVector<QColor> colors;
};

#endif // MODEL_LABELING_H
