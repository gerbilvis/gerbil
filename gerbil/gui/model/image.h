#ifndef MODEL_IMAGE_H
#define MODEL_IMAGE_H

#include <shared_data.h>
#include <multi_img_tasks.h>

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

class ImageModel : public QObject
{
	Q_OBJECT

	struct payload {
		payload(SharedMultiImgPtr i = SharedMultiImgPtr())
			: image(i), norm(MultiImg::NORM_OBSERVED), normRange(
				new SharedData<std::pair<multi_img::Value, multi_img::Value> >(
				  new std::pair<multi_img::Value, multi_img::Value>(0, 0)))
		{}

		// multispectral image data
		SharedMultiImgPtr image;

		// normalization mode and range
		MultiImg::NormMode normMode;
		data_range_ptr normRange;

		// cached single bands
		QMap<int, QPixmap> bands;
	};

public:
	explicit ImageModel(bool limitedMode);
	~ImageModel();

	const cv::Rect& getROI() { return roi; }

	/** @return dimensions of the image as a rectangle */
	cv::Rect loadImage(QString filename);
	void spawn(const cv::Rect& roi, bool reuse);
	
public slots:
	void computeBand(representation repr, int dim);
	void computeRGB();
	void postComputeRGB(bool success);

signals:
	void bandUpdate(QPixmap band, QString description);
	void rgbUpdate(QPixmap rgb);

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
};

#endif // MODEL_IMAGE_H
