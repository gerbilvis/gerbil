#ifndef MODEL_IMAGE_H
#define MODEL_IMAGE_H

#include "representation.h"
#include <shared_data.h>
#include <multi_img_util.h>
#include <multi_img_tasks.h>
#include <background_task_queue.h>

#include <QObject>
#include <QMap>
#include <QPixmap>
#include <vector>

class ImageModelPayload : public QObject {
	Q_OBJECT

public:
	/* always initialize image, as the SharedData will be passed around
	 * and used to enqueue tasks even before the image is created the
	 * first time. */
	ImageModelPayload(representation::t type)
		: type(type), image(new SharedMultiImgBase(new multi_img())),
		  normMode(MultiImg::NORM_OBSERVED), normRange(
			new SharedData<ImageDataRange> (
			  new ImageDataRange(0, 0)))
	{}

	// the type we have
	representation::t type;

	// multispectral image data
	SharedMultiImgPtr image;

	// normalization mode and range
	MultiImg::NormMode normMode;
	SharedDataRangePtr normRange;

	// cached single bands
	QMap<int, QPixmap> bands;

public slots:
	void processImageDataTaskFinished(bool success);
	void processDataRangeTaskFinished(bool success);

signals:
	void newImageData(representation::t type, SharedMultiImgPtr image);
	void dataRangeUpdate(representation::t type, ImageDataRange range);
};

class ImageModel : public QObject
{
	Q_OBJECT

public:
	typedef ImageModelPayload payload;

	explicit ImageModel(BackgroundTaskQueue &queue, bool limitedMode);
	~ImageModel();

	/** Return the number of bands in the input image.
	 *
	 * @note The number of bands in the current ROI image(s) may differ, see
	 * getNumBandsROI().
	 */
	// FIXME: rename getNumBandsFull() (altmann, jordan: ack)
	size_t getSize();
	/** Return the number of bands in the multispectral image that is currently
	 * used as ROI. */
	// FIXME 2013-06-19 altmann: This is assuming the number of bands is fixed for
	// all representations. Not sure if this is necessarily true.
	int getNumBandsROI();
	const cv::Rect& getROI() { return roi; }
	SharedMultiImgPtr getImage(representation::t type) { return map[type]->image; }
	SharedMultiImgPtr getFullImage() { return image_lim; }
	bool isLimitedMode() { return limitedMode; }

	// delete ROI information also in images
	void invalidateROI();

	/** @return dimensions of the image as a rectangle */
	cv::Rect loadImage(const std::string &filename);
	/** @arg bands number of bands needed (only effective for IMG type) */
	void spawn(representation::t type, const cv::Rect& roi, size_t bands = -1);

public slots:
	void computeBand(representation::t type, int dim);
	/** Compute rgb representation of full image.
	 *
	 * Emits fullRgbUpdate() when finished.
	 *
	 * @note Typically this is called once for each image, since the RGB representation
	 * for ROI-View does not need to be updated.
	 */
	void computeFullRgb();

//	/** Compute data range for given image representation.
//	 *
//	 * Emits dataRangeUdpate() after task has finished.
//	 */
//	void computeDataRange(representation::t type);

	void setNormalizationParameters(
			representation::t type,
			MultiImg::NormMode normMode,
			ImageDataRange targetRange);
signals:
	/** The data of the currently selected band has changed. */
	void bandUpdate(QPixmap band, QString description);

	void fullRgbUpdate(QPixmap fullRgb);
	void imageUpdate(representation::t type, SharedMultiImgPtr image);
	/** The data range for representation type has changed. */
	void dataRangeUdpate(representation::t type, const ImageDataRange& range);
private:
	// helper to spawn()
	bool checkProfitable(const cv::Rect& oldROI, const cv::Rect& newROI);

	// FIXME rename
	SharedMultiImgPtr image_lim; // big one
	// small ones (ROI) and their companion data:
	QMap<representation::t, payload*> map;

	// do we run in limited mode?
	bool limitedMode;

	// current region of interest
	cv::Rect roi;

	// TODO
	// int nBands;

	BackgroundTaskQueue &queue;

	/* we need to keep this guy around as we cannot release ownership from a
	 * shared_ptr.
	 */
	multi_img::ptr imgHolder;
};

#endif // MODEL_IMAGE_H
