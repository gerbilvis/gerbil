#ifndef MODEL_IMAGE_H
#define MODEL_IMAGE_H

#include "representation.h"
#include <shared_data.h>
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
			new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
			  new std::pair<multi_img::Value, multi_img::Value>(0, 0)))
	{}

	// the type we have
	representation::t type;

	// multispectral image data
	SharedMultiImgPtr image;

	// normalization mode and range
	MultiImg::NormMode normMode;
	data_range_ptr normRange;

	// cached single bands
	QMap<int, QPixmap> bands;

public slots:
	// connect the corresponding task's finishing with this slot
	void propagateFinishedCalculation(bool success);

signals:
	void newImageData(representation::t type, SharedMultiImgPtr image);
};

class ImageModel : public QObject
{
	Q_OBJECT

public:
	typedef ImageModelPayload payload;

	explicit ImageModel(BackgroundTaskQueue &queue, bool limitedMode);
	~ImageModel();

	size_t getSize();
	const cv::Rect& getROI() { return roi; }
	SharedMultiImgPtr getImage(representation::t type) { return map[type]->image; }

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

signals:
	void bandUpdate(QPixmap band, QString description);

	void fullRgbUpdate(QPixmap fullRgb);
	void imageUpdate(representation::t type, SharedMultiImgPtr image);

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

	BackgroundTaskQueue &queue;

	/* we need to keep this guy around as we cannot release ownership from a
	 * shared_ptr.
	 */
	multi_img::ptr imgHolder;
};

#endif // MODEL_IMAGE_H
