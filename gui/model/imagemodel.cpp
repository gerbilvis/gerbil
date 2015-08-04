#include "imagemodel.h"

#include "dialogs/openrecent/recentfile.h"

#include "../gerbil_gui_debug.h"
#include "gerbilapplication.h"

#include <background_task/tasks/cuda/gerbil_cuda_util.h>
#include <background_task/tasks/scopeimage.h>
#include <background_task/tasks/cuda/datarangecuda.h>
#include <background_task/tasks/cuda/gradientcuda.h>
#include <background_task/tasks/cuda/normrangecuda.h>
#include <background_task/tasks/tbb/band2qimagetbb.h>
#include <background_task/tasks/tbb/datarangetbb.h>
#include <background_task/tasks/tbb/gradienttbb.h>
#include <background_task/tasks/tbb/norml2tbb.h>
#include <background_task/tasks/tbb/normrangetbb.h>
#include <background_task/tasks/tbb/pcatbb.h>
#include <background_task/tasks/tbb/rescaletbb.h>
#include <background_task/tasks/tbb/rgbqttbb.h>

#include <multi_img/multi_img_offloaded.h>
#include <imginput.h>

#include <boost/make_shared.hpp>

#ifdef GERBIL_CUDA
	#include <opencv2/gpu/gpu.hpp>
	#define USE_CUDA_GRADIENT
//	#define USE_CUDA_DATARANGE
//	#define USE_CUDA_CLAMP
#endif

ImageModel::ImageModel(BackgroundTaskQueue &queue, bool lm, QObject *parent)
	: QObject(parent), limitedMode(lm), queue(queue),
	  image_lim(new SharedMultiImgBase(new multi_img())),
	  nBands(0), nBandsOld(0)
{
	foreach (representation::t i, representation::all()) {
		map.insert(i, new payload(i));
	}

	foreach (payload *p, map) {
		connect(p, SIGNAL(newImageData(representation::t,SharedMultiImgPtr)),
				this,
				SLOT(processNewImageData(representation::t,SharedMultiImgPtr)));
		connect(p, SIGNAL(dataRangeUpdate(representation::t,multi_img::Range)),
				this,
				SIGNAL(observedDataRangeUdpate(representation::t,
											   multi_img::Range)));
	}
}

ImageModel::~ImageModel()
{
	foreach (payload *p, map)
		delete p;
}

int ImageModel::getNumBandsFull()
{
	SharedMultiImgBaseGuard guard(*image_lim);
	return image_lim->getBase().size();
}

int ImageModel::getNumBandsROI()
{
	return nBands;
}

cv::Rect ImageModel::getFullImageRect()
{
	SharedDataLock lock(image_lim->mutex);
	cv::Rect dims(0, 0,
				  image_lim->getBase().width, image_lim->getBase().height);
	return dims;
}

cv::Rect ImageModel::loadImage(const QString &filename)
{
	// do a more complicated transformation to preserve non-ascii filenames
	std::string fn = filename.toLocal8Bit().constData();
	if (limitedMode) {
		// create offloaded image
		std::pair<std::vector<std::string>, std::vector<multi_img::BandDesc> >
				filelist = multi_img::parse_filelist(fn);
		image_lim = boost::make_shared<SharedMultiImgBase>
				(new multi_img_offloaded(filelist.first, filelist.second));
	} else {
		// create using ImgInput
		imginput::ImgInputConfig inputConfig;
		inputConfig.file = fn;
		multi_img::ptr img = imginput::ImgInput(inputConfig).execute();
		image_lim = boost::make_shared<SharedMultiImgBase>(img);
	}

	multi_img_base &i = image_lim->getBase();
	if (i.empty()) {
		GerbilApplication::instance()->
			userError("Image file could not be read.");
		return cv::Rect();
	}

	if (i.size() < 3) {
		GerbilApplication::instance()->
			userError("Image unsupported: Contains less than three bands (channels).");
		return cv::Rect();
	} else {
		// Update recent files list.
		QPixmap pvpm = fullRgb();
		RecentFile::appendToRecentFilesList(filename, pvpm);
		return cv::Rect(0, 0, i.width, i.height);
	}
}

void ImageModel::invalidateROI()
{
	// set roi to empty rect
	roi = cv::Rect();
	foreach (payload *p, map) {
		if (!p->image)
			continue;

		SharedDataLock lock(p->image->mutex);
		(*(p->image))->roi = roi;
	}
}

void ImageModelPayload::processImageDataTaskFinished(bool success)
{
	if (!success)
		return;
//	std::cout << " type " << this->type
//			  << " minval " << (*image)->minval
//			  << " maxval " << (*image)->maxval << std::endl;
	// signal new image data
	emit newImageData(type, image);
	emit dataRangeUpdate(type, **normRange);

}

void ImageModel::spawn(representation::t type, const cv::Rect &newROI, int bands)
{
	// Store previous state.
	// IMG is guaranteed to be self-subscribed by controller and thus we always
	// get this update when a new ROI is spawned.
	if (representation::IMG == type) {
		nBandsOld = nBands;
		oldRoi = roi;
	}

	GGDBGM(type << " oldRoi "<< oldRoi << " newROI " << newROI << endl);

	// one ROI for all representations, effectively
	roi = newROI;

	// shortcuts for convenience
	SharedMultiImgPtr image = map[representation::IMG]->image;
	SharedMultiImgPtr imagenorm = map[representation::NORM]->image;
	SharedMultiImgPtr gradient = map[representation::GRAD]->image;
	SharedMultiImgPtr imagepca = map[representation::IMGPCA]->image;
//	SharedMultiImgPtr gradpca = map[representation::GRADPCA]->image;

	// scoping and spectral rescaling done for IMG
	if (type == representation::IMG) {
		// scope image to new ROI
		SharedMultiImgPtr scoped_image(new SharedMultiImgBase(NULL));
		BackgroundTaskPtr taskScope(new ScopeImage(
			image_lim, scoped_image, roi));
		queue.push(taskScope);

		// sanitize spectral rescaling parameters
		assert(getNumBandsFull() > 0);
		if ( (bands < 1 && getNumBandsROI() < 1) // no ROI yet
			 || bands > getNumBandsFull()) { // bands too large
			bands = getNumBandsFull();
		} else if (bands < 1) {
			// default, use nbands from ROI
			bands = getNumBandsROI();
		} else if (bands <= 2) { // to few bands
			bands = 3;
		}

		// perform spectral rescaling
		BackgroundTaskPtr taskRescale(new RescaleTbb(
			scoped_image, image, bands));
		queue.push(taskRescale);
	} 

   // NORM / GRAD
   if (type == representation::NORM) {
		BackgroundTaskPtr taskNorm(new NormL2Tbb(
			image, imagenorm));
		queue.push(taskNorm);
	} else  if (type == representation::GRAD) {
#ifdef USE_CUDA_GRADIENT
		if (haveCvCudaGpu()) {
			BackgroundTaskPtr taskGradient(new GradientCuda(
				image, gradient));
			queue.push(taskGradient);
		} else {
#else
		{
#endif
			BackgroundTaskPtr taskGradient(new GradientTbb(
				image, gradient));
			queue.push(taskGradient);
		}
	}

	// user-customizable norm range calculation, sets minval/maxval of the image
	if (type == representation::IMG || type == representation::GRAD)
	{
		SharedMultiImgPtr target = map[type]->image;
		SharedMultiImgRangePtr range = map[type]->normRange;
		multi_img::NormMode mode =  map[type]->normMode;
		// TODO: a small hack in NormRangeTBB to determine theoretical range
		int isGRAD = (type == representation::GRAD ? 1 : 0);

		SharedDataLock hlock(range->mutex);
		//GGDBGM(format("%1% range %2%") %type %**range << endl);
		double min = (*range)->min;
		double max = (*range)->max;
		hlock.unlock();
#ifdef USE_CUDA_DATARANGE
		if (haveCvCudaGpu()) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				target, range, mode, isGRAD, min, max, true));
			queue.push(taskNormRange);
		} else {
#else
		{
#endif
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				target, range, mode, isGRAD, min, max, true));
			queue.push(taskNormRange);
		}
	} 
	
	// IMGPCA / GRADPCA
    if (type == representation::IMGPCA && imagepca.get()) {
		BackgroundTaskPtr taskPca(new PcaTbb(
			image, imagepca, 10));
		queue.push(taskPca);
/*	} else if (type == representation::GRADPCA && gradpca.get()) {
		BackgroundTaskPtr taskPca(new PcaTbb(
			gradient, gradpca, 0));
		queue.push(taskPca);*/
	}

	// emit signal after all tasks are finished and fully updated data available
	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)),
					 map[type], SLOT(processImageDataTaskFinished(bool)));
	queue.push(taskEpilog);
}

void ImageModel::respawn(representation::t type)
{
	if ((*map[type]->image)->empty()) {
		std::cerr << "ImageModel::respawn(): ignored, no payload available."
					 << std::endl;
		return;
	}
	emit imageUpdate(type, map[type]->image, /* duplicate */ true);
}

void ImageModel::computeBand(representation::t type, int dim)
{
	//GGDBGM(type << " " << dim << endl);
	QMap<int, QPixmap> &m = map[type]->bands;
	SharedMultiImgPtr src = map[type]->image;
	assert(src);

	SharedDataLock hlock(src->mutex);

	// ensure sane input
	int size = (*src)->size();
	if (dim >= size)
		dim = 0;

	// retrieve wavelength metadata
	std::string banddesc;
	if((*src)->meta.size() > 0) {
		banddesc = (*src)->meta[dim].str();
	}
	hlock.unlock();

	// compute image data if necessary
	if (!m.contains(dim)) {
		SharedMultiImgPtr src = map[type]->image;
		qimage_ptr dest(new SharedData<QImage>(new QImage()));

		SharedDataLock hlock(src->mutex);
		BackgroundTaskPtr taskConvert(
					new Band2QImageTbb(src, dest, dim));
		taskConvert->run();
		hlock.unlock();

		m[dim] = QPixmap::fromImage(**dest);
	}

	QString desc;
	QString typestr = representation::prettyString(type);
	if (banddesc.empty())
		desc = QString("%1 Band #%2").arg(typestr).arg(dim+1);
	else
		desc = QString("%1 Band %2").arg(typestr).arg(banddesc.c_str());

	emit bandUpdate(type, dim, m[dim], desc);
}

void ImageModel::computeFullRgb()
{
	emit fullRgbUpdate(fullRgb());
}

void ImageModel::setNormalizationParameters(representation::t type,
		multi_img::NormMode normMode,
		multi_img_base::Range targetRange)
{
	//GGDBGM(type << " " << targetRange << endl);
	map[type]->normMode = normMode;
	SharedDataLock lock(map[type]->normRange->mutex);
	**(map[type]->normRange) = targetRange;
}


void ImageModel::processNewImageData(representation::t type,
									 SharedMultiImgPtr image)
{
	// invalidate band caches
	map[type]->bands.clear();

	if (representation::IMG == type) {
		SharedDataLock lock(image->mutex);
		nBands = (*image)->size();
	} else if (representation::GRAD == type) {
		// check consistency of gradient
		assert((*image)->size() == nBands-1);
	}
	if (nBandsOld != nBands) {
		emit numBandsROIChanged(nBands);
	}
	if (representation::IMG == type && oldRoi != roi) {
		//GGDBGM("oldRoi "<< oldRoi << " cur roi " << roi << endl);
		emit roiRectChanged(roi);
	}
	emit imageUpdate(type, image, /* duplicate */ false);
}

QPixmap ImageModel::fullRgb()
{
	// FIXME Caching

	qimage_ptr fullRgb(new SharedData<QImage>(NULL));
	/* we do it instantly as this is typically what the user wants to see first,
	 * and not wait for it while the queue processes other things */
	BackgroundTaskPtr taskRgb(new RgbTbb(
		image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)),
								  fullRgb));
	taskRgb->run();

	QPixmap p = QPixmap::fromImage(**fullRgb);
	return p;
}
