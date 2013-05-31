#include "model/image.h"
#include <multi_img_offloaded.h>
#include <boost/make_shared.hpp>

ImageModel::ImageModel(bool lm)
	: limitedMode(lm),
	  image_lim(new SharedMultiImgBase(new multi_img())),
	  full_rgb(new SharedData<QImage>(new QImage()))
{
	/* TODO: previously, image and grad were initialized here. make sure it is
	 * o.k. that we don't do it anymore.
	 */
	for (int i = 0; i < REPSIZE; ++i) {
		map.insert((representation)i, new payload());
	}
}

ImageModel::~ImageModel()
{
	foreach (payload *p, map)
		delete p;
}

// empty deleter to prevent sharedptr from deleting when we transfer ownership
void Deleter(multi_img* ptr)
{
	std::cerr << "debug: not deleting image of size " << ptr->width
			  << ", " << ptr->height << ", " << ptr->size() << std::endl;
	return;
}

cv::Rect ImageModel::loadImage(QString filename)
{
	if (limited_mode) {
		// create offloaded image
		image_lim = boost::make_shared<SharedMultiImgBase>(
					new	multi_img_offloaded(filelist.first, filelist.second));
	} else {
		// create using ImgInput
		vole::ImgInputConfig inputConfig;
		inputConfig.file = filename;
		multi_img::ptr img = vole::ImgInput(inputConfig).execute();

		// transfer ownership
		multi_img::ptr img2(new multi_img(), Deleter);
		img.swap(img2); // now shared pointer without delete owns the image
		image_lim = boost::make_shared<SharedMultiImgBase>(img2.get());
		// SharedData object will eventually delete the image
	}

	if ((*image_lin)->empty()) {
		return cv::Rect;
	} else {
		return cv::Rect(0, 0, (*image_lin)->width, (*image_lin)->height);
	}
}

void ImageModel::spawn(const cv::Rect &newROI, bool reuse)
{
	// reset ROI if not reuse
	if (!reuse) {
		cv::Rect empty(0, 0, 0, 0);
		foreach (payload *p, map) {
			if (!p->image)
				continue;

			SharedDataLock lock(p->image->mutex);
			(*(p->image))->roi = empty;
		}
	}

	// prepare incremental update and test worthiness
	std::vector<cv::Rect> sub, add;
	if (reuse) {
		/* compute if it is profitable to add/sub pixels given old and new ROI,
		 * instead of full recomputation, and retrieve corresponding regions
		 */
		bool profitable = MultiImg::Auxiliary::rectTransform(roi, newROI,
															 sub, add);

		if (!profitable)
			reuse = false;
	}

	// set new ROI
	roi = newROI;
	// TODO: others do not know about it yet

	sets_ptr tmp_sets_image(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_imagepca(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_gradient(new SharedData<std::vector<BinSet> >(NULL));
	sets_ptr tmp_sets_gradientpca(new SharedData<std::vector<BinSet> >(NULL));
	if (reuse) {
		viewerContainer->subImage(IMG, tmp_sets_image, sub, roi);
		viewerContainer->subImage(IMG, tmp_sets_gradient, sub, roi);
		viewerContainer->subImage(IMG, tmp_sets_imagepca, sub, roi);
		viewerContainer->subImage(IMG, tmp_sets_gradientpca, sub, roi);
	}

	updateRGB(true);
	rgbDock->setEnabled(true);

	// TODO: all done by lm->updateRoi
	labels = cv::Mat1s(full_labels, roi);
	bandView->labels = labels;
	viewerContainer->setLabels(labels);
	// end TODO

	// TODO: we better test here if numbands changed, instead of just relying on
	// what the signal sender told us
	size_t numbands;
	{
		SharedMultiImgBaseGuard guard(*image_lim);
		numbands = bandsSlider->value();
		if (numbands <= 2)
			numbands = 3;
		if (numbands > (*image_lim)->size())
			numbands = (*image_lim)->size();
	}

	SharedMultiImgPtr scoped_image(new SharedMultiImgBase(NULL));
	BackgroundTaskPtr taskScope(new MultiImg::ScopeImage(
		image_lim, scoped_image, roi));
	queue.push(taskScope);

	// invalidate band caches
	foreach (payload *p, map) {
		p->bands.clear();
	}

	// TODO: only do when sth. changes?
	BackgroundTaskPtr taskRescale(new MultiImg::RescaleTbb(
		scoped_image, image, numbands, roi));
	queue.push(taskRescale);

	{
		SharedDataLock hlock(normIMGRange->mutex);
		double min = (*normIMGRange)->first;
		double max = (*normIMGRange)->second;
		hlock.unlock();

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskImgNormRange(new NormRangeCuda(
				image, normIMGRange, normIMG, 0, min, max, true, roi));
			queue.push(taskImgNormRange);
		} else {
			BackgroundTaskPtr taskImgNormRange(new NormRangeTbb(
				image, normIMGRange, normIMG, 0, min, max, true, roi));
			queue.push(taskImgNormRange);
		}
	}

	if (reuse) {
		viewerContainer->addImage(IMG, tmp_sets_image, add, roi);
	} else {
		viewerContainer->setImage(IMG, image, roi);
	}

	BackgroundTaskPtr taskImgFinish(new BackgroundTask(roi));
	QObject::connect(taskImgFinish.get(), SIGNAL(finished(bool)),
		viewerContainer, SLOT(imgCalculationComplete(bool)), Qt::QueuedConnection);
	queue.push(taskImgFinish);

	if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_GRADIENT) {
		BackgroundTaskPtr taskGradient(new MultiImg::GradientCuda(
			image, gradient, roi));
		queue.push(taskGradient);
	} else {
		BackgroundTaskPtr taskGradient(new MultiImg::GradientTbb(
			image, gradient, roi));
		queue.push(taskGradient);
	}

	{
		SharedDataLock hlock(normGRADRange->mutex);
		double min = (*normGRADRange)->first;
		double max = (*normGRADRange)->second;
		hlock.unlock();

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskGradNormRange(new NormRangeCuda(
				gradient, normGRADRange, normGRAD, 1, min, max, true, roi));
			queue.push(taskGradNormRange);
		} else {
			BackgroundTaskPtr taskGradNormRange(new NormRangeTbb(
				gradient, normGRADRange, normGRAD, 1, min, max, true, roi));
			queue.push(taskGradNormRange);
		}
	}

	if (reuse) {
		viewerContainer->addImage(GRAD, tmp_sets_gradient, add, roi);
	} else {
		viewerContainer->setImage(GRAD, gradient, roi);
	}

	BackgroundTaskPtr taskGradFinish(new BackgroundTask(roi));
	QObject::connect(taskGradFinish.get(), SIGNAL(finished(bool)),
		viewerContainer, SLOT(gradCalculationComplete(bool)), Qt::QueuedConnection);
	queue.push(taskGradFinish);

	if (imagepca.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			image, imagepca, 0, roi));
		queue.push(taskPca);

		if (reuse) {
			viewerContainer->addImage(IMGPCA, tmp_sets_imagepca, add, roi);
		} else {
			viewerContainer->setImage(IMGPCA, imagepca, roi);
		}

		BackgroundTaskPtr taskImgPcaFinish(new BackgroundTask(roi));
		QObject::connect(taskImgPcaFinish.get(), SIGNAL(finished(bool)),
			viewerContainer, SLOT(imgPcaCalculationComplete(bool)), Qt::QueuedConnection);
		queue.push(taskImgPcaFinish);
	}

	if (gradientpca.get()) {
		BackgroundTaskPtr taskPca(new MultiImg::PcaTbb(
			gradient, gradientpca, 0, roi));
		queue.push(taskPca);

		if (reuse) {
			viewerContainer->addImage(GRADPCA, tmp_sets_gradientpca, add, roi);
		} else {
			viewerContainer->setImage(GRADPCA, gradientpca, roi);
		}

		BackgroundTaskPtr taskGradPcaFinish(new BackgroundTask(roi));
		QObject::connect(taskGradPcaFinish.get(), SIGNAL(finished(bool)),
			viewerContainer, SLOT(gradPcaCalculationComplete(bool)), Qt::QueuedConnection);
		queue.push(taskGradPcaFinish);
	}
}

void ImageModel::computeBand(representation repr, int dim)
{
	QMap<int, QPixmap> &m = map[repr]->bands;

	if (!m.contains(dim)) {
		SharedMultiImgPtr src = map[repr]->image;
		SharedDataLock hlock(src->mutex);

		qimage_ptr qimg(new SharedData<QImage>(new QImage()));

		BackgroundTaskPtr taskConvert(
					new MultiImg::Band2QImageTbb(multi, qimg, dim));
		taskConvert->run();

		hlock.unlock();

		m[dim] = new QPixmap(QPixmap::fromImage(**qimg));
	}

	// retrieve wavelength information
	SharedMultiImgPtr src = map[repr]->image;
	SharedDataLock hlock(src->mutex);
	std::string banddesc = (*src)->meta[dim].str();
	hlock.unlock();
	QString desc;
	if (banddesc.empty())
		desc = QString("%1 Band #%2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO covers only IMG/GRAD
			.arg(dim+1);
	else
		desc = QString("%1 Band %2")
			.arg(type == GRAD ? "Gradient" : "Image") // TODO
			.arg(banddesc.c_str());

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
