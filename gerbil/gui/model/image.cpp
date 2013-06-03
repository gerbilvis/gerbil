#include "model/image.h"
#include "tasks/normrangecuda.h"
#include "tasks/normrangetbb.h"
#include "tasks/rgbtbb.h"

#include <multi_img_offloaded.h>
#include <imginput.h>
#include <boost/make_shared.hpp>

#include <opencv2/gpu/gpu.hpp>

#define USE_CUDA_GRADIENT       1
#define USE_CUDA_DATARANGE      0
#define USE_CUDA_CLAMP          0

ImageModel::ImageModel(BackgroundTaskQueue &queue, bool lm)
	: limitedMode(lm), queue(queue),
	  image_lim(new SharedMultiImgBase(new multi_img())),
	  full_rgb(new SharedData<QImage>(new QImage()))
{
	for (int i = 0; i < REPSIZE; ++i) {
		representations[i] = (representation)i;
		map.insert((representation)i, new payload((representation)i));
	}

	foreach (payload *p, map) {
		connect(p, SIGNAL(newImageData(representation,SharedMultiImgPtr)),
				this, SIGNAL(imageUpdate(representation,SharedMultiImgPtr)));
	}
}

ImageModel::~ImageModel()
{
	foreach (payload *p, map)
		delete p;

	if (!limitedMode) {
		// release image_lim as imgHolder has still ownership and will delete
		image_lim->swap((multi_img_base*)0);
	}
}

size_t ImageModel::getSize()
{
	SharedMultiImgBaseGuard guard(*image_lim);
	return (*image_lim)->size();
}

cv::Rect ImageModel::loadImage(const std::string &filename)
{
	if (limitedMode) {
		// create offloaded image
		std::pair<std::vector<std::string>, std::vector<multi_img::BandDesc> >
				filelist = multi_img::parse_filelist(filename);
		image_lim = boost::make_shared<SharedMultiImgBase>
				(new multi_img_offloaded(filelist.first, filelist.second));
	} else {
		// create using ImgInput
		vole::ImgInputConfig inputConfig;
		inputConfig.file = filename;
		imgHolder = vole::ImgInput(inputConfig).execute();
		image_lim = boost::make_shared<SharedMultiImgBase>(imgHolder.get());
	}

	if ((*image_lim)->empty()) {
		return cv::Rect();
	} else {
		return cv::Rect(0, 0, (*image_lim)->width, (*image_lim)->height);
	}
}

void ImageModel::invalidateROI()
{
	roi = cv::Rect();
	foreach (payload *p, map) {
		if (!p->image)
			continue;

		SharedDataLock lock(p->image->mutex);
		(*(p->image))->roi = roi;
	}
}

void ImageModel::payload::propagateFinishedCalculation(bool success)
{
	if (!success)
		return;

	// signal new image data
	emit newImageData(type, image);
}

void ImageModel::spawn(representation type, const cv::Rect &newROI, size_t bands)
{
	// one ROI for all, effectively
	roi = newROI;

	// invalidate band caches
	map[IMG]->bands.clear();

	// scoping and spectral rescaling done for IMG
	if (type == IMG) {
		// scope image to new ROI
		SharedMultiImgPtr scoped_image(new SharedMultiImgBase(NULL));
		BackgroundTaskPtr taskScope(new MultiImg::ScopeImage(
			image_lim, scoped_image, roi));
		queue.push(taskScope);

		// sanitize spectral rescaling parameters
		size_t fullbands = getSize();
		if (bands == -1 || bands > fullbands)
			bands = fullbands;
		if (bands <= 2)
			bands = 3;

		SharedMultiImgPtr image = map[IMG]->image;

		// perform spectral rescaling
		BackgroundTaskPtr taskRescale(new MultiImg::RescaleTbb(
			scoped_image, image, bands, roi));
		queue.push(taskRescale);
	}

	if (type == GRAD) {
		SharedMultiImgPtr image = map[IMG]->image, gradient = map[GRAD]->image;

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_GRADIENT) {
			BackgroundTaskPtr taskGradient(new MultiImg::GradientCuda(
				image, gradient, roi));
			queue.push(taskGradient);
		} else {
			BackgroundTaskPtr taskGradient(new MultiImg::GradientTbb(
				image, gradient, roi));
			queue.push(taskGradient);
		}
	}

	// user-customizable norm range calculation, sets minval/maxval of the image
	if (type == IMG || type == GRAD)
	{
		SharedMultiImgPtr target = map[type]->image;
		data_range_ptr range = map[type]->normRange;
		MultiImg::NormMode mode =  map[type]->normMode;
		// TODO: a small hack in NormRangeTBB to determine theoretical range
		int isGRAD = (type == GRAD ? 1 : 0);

		SharedDataLock hlock(range->mutex);
		double min = (*range)->first;
		double max = (*range)->second;
		hlock.unlock();

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				target, range, mode, isGRAD, min, max, true, roi));
			queue.push(taskNormRange);
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				target, range, mode, isGRAD, min, max, true, roi));
			queue.push(taskNormRange);
		}
	}


	if (type == IMGPCA && map[IMGPCA]->image.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			map[IMG]->image, map[IMGPCA]->image, 0, roi));
		queue.push(taskPca);
	}

	if (type == GRADPCA && map[GRADPCA]->image.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			map[GRAD]->image, map[GRADPCA]->image, 0, roi));
		queue.push(taskPca);
	}

	// emit signal after all tasks are finished and fully updated data available
	BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
					 map[type], SLOT(propagateFinishedCalculation(bool)));
	queue.push(taskEpilog);
}

void ImageModel::computeBand(representation type, int dim)
{
	QMap<int, QPixmap> &m = map[type]->bands;

	if (!m.contains(dim)) {
		SharedMultiImgPtr src = map[type]->image;
		qimage_ptr dest(new SharedData<QImage>(new QImage()));

		SharedDataLock hlock(src->mutex);
		BackgroundTaskPtr taskConvert(
					new MultiImg::Band2QImageTbb(src, dest, dim));
		taskConvert->run();
		hlock.unlock();

		m[dim] = QPixmap::fromImage(**dest);
	}

	// retrieve wavelength information
	SharedMultiImgPtr src = map[type]->image;
	SharedDataLock hlock(src->mutex);
	std::string banddesc = (*src)->meta[dim].str();
	hlock.unlock();
	QString desc;
	const char * const str[] =
		{ "Image", "Gradient", "Image PCA", "Gradient PCA" };
	if (banddesc.empty())
		desc = QString("%1 Band #%2").arg(str[type]).arg(dim+1);
	else
		desc = QString("%1 Band %2").arg(str[type]).arg(banddesc.c_str());

	emit bandUpdate(m[dim], desc);
}

void ImageModel::computeRGB() // TODO: bool instantly
{
	SharedDataLock hlock(full_rgb->mutex);

	/* we do it instantly as this is typically what the user wants to see first,
	 * and not wait for it while the queue processes other things */
	BackgroundTaskPtr taskRgb(new RgbTbb(
		image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
								  full_rgb));
	taskRgb->run();
	hlock.unlock();
	postComputeRGB(true);
}

void ImageModel::postComputeRGB(bool success)
{
	if (!success)
		return;

	SharedDataLock hlock(full_rgb->mutex);
	QPixmap ret = QPixmap::fromImage(**full_rgb);
	hlock.unlock();
	emit rgbUpdate(ret);
}

/* only used for debugging */
std::ostream &operator <<(std::ostream &os, const representation &r)
{
	assert(0 <= r);
	assert(r < REPSIZE);
	const char * const str[] = { "IMG", "GRAD", "IMGPCA", "GRADPCA" };
	os << str[r];
}
