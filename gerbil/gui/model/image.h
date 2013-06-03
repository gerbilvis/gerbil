#ifndef MODEL_IMAGE_H
#define MODEL_IMAGE_H

#include <shared_data.h>
#include <multi_img_tasks.h>
#include <background_task_queue.h>

#include <QObject>
#include <QMap>
#include <QPixmap>
#include <vector>

// TODO: put in ImageModel namespace?
enum representation {
	IMG = 0,
	GRAD = 1,
	IMGPCA = 2,
	GRADPCA = 3,
	REPSIZE = 4
	// if you add something, also change operator<<!
};

// representation in debug output
std::ostream &operator<<(std::ostream& os, const representation& r);

class ImageModelPayload : public QObject {
	Q_OBJECT

public:
	/* always initialize image, as the SharedData will be passed around
	 * and used to enqueue tasks even before the image is created the
	 * first time. */
	ImageModelPayload(representation type)
		: type(type), image(new SharedMultiImgBase(new multi_img())),
		  normMode(MultiImg::NORM_OBSERVED), normRange(
			new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
			  new std::pair<multi_img::Value, multi_img::Value>(0, 0)))
	{}

	// the type we have
	representation type;

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
	void newImageData(representation type, SharedMultiImgPtr image);
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
	SharedMultiImgPtr getImage(representation type) { return map[type]->image; }

	// delete ROI information also in images
	void invalidateROI();

	/** @return dimensions of the image as a rectangle */
	cv::Rect loadImage(const std::string &filename);
	/** @arg bands number of bands needed (only effective for IMG type) */
	void spawn(representation type, const cv::Rect& roi, size_t bands = -1);
	
	// for easy looping over all valid representations
	QMap<int, representation> representations;

public slots:
	void computeBand(representation type, int dim);
	void computeRGB();
	void postComputeRGB(bool success);

signals:
	void bandUpdate(QPixmap band, QString description);
	void rgbUpdate(QPixmap rgb);
	void imageUpdate(representation type, SharedMultiImgPtr image);

private:
	// helper to spawn()
	bool checkProfitable(const cv::Rect& oldROI, const cv::Rect& newROI);

	// FIXME rename
	SharedMultiImgPtr image_lim; // big one
	// small ones (ROI) and their companion data:
	QMap<representation, payload*> map;

	// do we run in limited mode?
	bool limitedMode;

	// current region of interest
	cv::Rect roi;

	// rgb representation to be used for ROI selection
	/* this is CMF currently, could be something else later on */
	qimage_ptr full_rgb;

	BackgroundTaskQueue &queue;

	/* we need to keep this guy around as we cannot release ownership from a
	 * shared_ptr.
	 */
	multi_img::ptr imgHolder;
};

#endif // MODEL_IMAGE_H
