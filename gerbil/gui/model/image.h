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
	// This slot is connected to the epilog task in Image::spawn() and in turn
	// emits the signals newImageData() and dataRangeUpdate() in this order.
	void processImageDataTaskFinished(bool success);

signals:
	// newImageData() and dataRangeUpdate are availabe to ImageModel clients
	// as ImageModel::imageUpdate() and ImageModel::dataRangeUdpate().
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
	int getNumBandsFull();

	/** Return the number of bands in the multispectral image that is currently
	 * used as ROI. */
	int getNumBandsROI();

	/** Returns the Region of Interest (ROI) of the stored image representations. */
	const cv::Rect& getROI() { return roi; }

	/** Returns a SharedMultiImgPtr to the image data with representation type.
	 *
	 * This is limited to the current ROI. The referenced SharedMultiImgBase
	 * object will remain in-place during runtime. The multi_img managed by it
	 * will be replaced on ROI changes and other operations.  The
	 * SharedMultiImgBase provides a mutex that needs to be locked for
	 * concurrent access.
	 */
	SharedMultiImgPtr getImage(representation::t type) { return map[type]->image; }

	/** Returns a SharedMultiImgPtr to the input image data.
	 *
	 * Although the image data may be modified, the referenced
	 * SharedMultiImgBase will remain in-place during runtime.
	 * SharedMultiImgBase provides a mutex that needs to be locked for
	 * concurrent access.
	 */
	SharedMultiImgPtr getFullImage() { return image_lim; }

	/** Return the dimensions of the full (input) image as a cv::Rect.
	 *
	 * @note Be aware that this function locks the SharedData mutex.
	 */
	cv::Rect getFullImageRect();

	bool isLimitedMode() { return limitedMode; }

	// delete ROI information also in images
	void invalidateROI();

	/** @return dimensions of the image as a rectangle */
	cv::Rect loadImage(const std::string &filename);
	/** @arg bands number of bands needed (only effective for IMG type) */
	void spawn(representation::t type, const cv::Rect& roi, int bands = -1);

public slots:
	void computeBand(representation::t type, int dim);
	/** Compute rgb representation of full image.
	 *
	 * Emits fullRgbUpdate() when finished.
	 *
	 * @note Typically this is called only once upon loading, since the RGB
	 * representation for ROI-View does not need to be updated.
	 */
	void computeFullRgb();

	void setNormalizationParameters(
			representation::t type,
			MultiImg::NormMode normMode,
			ImageDataRange targetRange);

signals:
	/** The data of the currently selected band has changed. */
	void bandUpdate(representation::t repr, int bandId,
					QPixmap band, QString description);

	void fullRgbUpdate(QPixmap fullRgb);

	/** The ROI image data for representation type has changed.
	 *
	 * This signal is emitted whenever the ROI is set to a new rect or the
	 * underlying data has been altered. Basically it is emitted after every
	 * spawn().
	 * See dataRangeUdpate, numBandsROIChanged and roiRectChanged for more
	 * specific signals.
	 */
	void imageUpdate(representation::t type, SharedMultiImgPtr image);

	/** The data range for representation type has changed. */
	void dataRangeUdpate(representation::t type, const ImageDataRange& range);

	/** The number of spectral bands of the ROI image has changed to nBands. 
	 * 
	 * numBandsROIChanged() is only emitted if the number of spectral bands has
	 * actually changed. That is, there will be no signal if the ROI image data
	 * has been altered while keeping the number of bands unchanged. 
	 */
	void numBandsROIChanged(int nBands);

	/** A new ROI has been applied.
	 *
	 * roiRectChanged() is emitted when the rect of the ROI changes,
	 * independently of wheter the image data in the ROI has changed.
	 */
	void roiRectChanged(cv::Rect roi);

protected slots:
	// payload background task has finished
	void processNewImageData(representation::t type, SharedMultiImgPtr image);
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

	// previous region of interest
	cv::Rect oldRoi;

	// current number of spectral bands in the IMG representation
	int nBands;

	// previous number of spectral bands in the IMG representation
	int nBandsOld;

	BackgroundTaskQueue &queue;

	/* we need to keep this guy around as we cannot release ownership from a
	 * shared_ptr.
	 */
	multi_img::ptr imgHolder;
};

#endif // MODEL_IMAGE_H
