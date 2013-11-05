/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "widgets/mainwindow.h"
#include "controller/controller.h"
#include "controller/distviewcontroller.h"
#include "iogui.h"
#include "commandrunner.h"
/*#include "tasks/rgbtbb.h"
#include "tasks/normrangecuda.h"
#include "tasks/normrangetbb.h"
#include <background_task/background_task_queue.h>
*/

#include "docks/illumdock.h"
#include "docks/clusteringdock.h"

#include <labeling.h>
#include <qtopencv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <QPainter>
#include <QIcon>
#include <QSignalMapper>
#include <iostream>
#include <QShortcut>
#include <QFileInfo>

MainWindow::MainWindow(bool limitedMode)
	: limitedMode(limitedMode),
	  contextMenu(NULL)
{
	// create all objects
	setupUi(this);
}

void MainWindow::initUI(std::string filename)
{
	/* set title */
	QFileInfo fi(QString::fromStdString(filename));
	setWindowTitle(QString("Gerbil - %1").arg(fi.completeBaseName()));
}

void MainWindow::initSignals(Controller *chief, DistViewController *chief2)
{
	/* slots & signals: GUI only */
	connect(docksButton, SIGNAL(clicked()),
			this, SLOT(openContextMenu()));

//	we decided to remove this functionality for now
//	connect(bandDock, SIGNAL(topLevelChanged(bool)),
//			this, SLOT(reshapeDock(bool)));

	/* buttons to alter label display dynamics */
	connect(ignoreButton, SIGNAL(toggled(bool)),
			markButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			nonmarkButton, SLOT(setDisabled(bool)));
	connect(ignoreButton, SIGNAL(toggled(bool)),
			singleButton, SLOT(setDisabled(bool)));

	connect(ignoreButton, SIGNAL(toggled(bool)),
			chief, SIGNAL(toggleIgnoreLabels(bool)));
	connect(singleButton, SIGNAL(toggled(bool)),
			chief, SIGNAL(toggleSingleLabel(bool)));

	// label manipulation from current dist_view
	connect(addButton, SIGNAL(clicked()),
			chief2, SLOT(addHighlightToLabel()));
	connect(remButton, SIGNAL(clicked()),
			chief2, SLOT(remHighlightFromLabel()));

	connect(markButton, SIGNAL(toggled(bool)),
			chief2, SIGNAL(toggleLabeled(bool)));
	connect(nonmarkButton, SIGNAL(toggled(bool)),
			chief2, SIGNAL(toggleUnlabeled(bool)));

//	connect(chief2, SIGNAL(normTargetChanged(bool)),
//			this, SLOT(normTargetChanged(bool)));

	/// global shortcuts
	QShortcut *scr = new QShortcut(Qt::CTRL + Qt::Key_S, this);
	connect(scr, SIGNAL(activated()), this, SLOT(screenshot()));
}

void MainWindow::addDistView(QWidget *frame)
{
	// determine position before spacer (last element in the layout)
	int pos = distviewLayout->count() - 1;
	// add with stretch = 1 so they will stay evenly distributed in space
	distviewLayout->insertWidget(pos, frame, 1);

	/* TODO: the spacer which is now in the .ui would be added like this
	 * previously. If current method fails, reconsider doing this: */
	// vLayout->addStretch(); // align on top when all folded
}

void MainWindow::setGUIEnabled(bool enable, TaskType tt)
{
	ignoreButton->setEnabled(enable || tt == TT_TOGGLE_LABELS);
	addButton->setEnabled(enable);
	remButton->setEnabled(enable);

	// TODO -> NormDock
//	normDock->setEnabled((enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD) && !limitedMode);
//	normIButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_IMG);
//	normGButton->setEnabled(enable || tt == TT_NORM_RANGE || tt == TT_CLAMP_RANGE_GRAD);
//	normModeBox->setEnabled(enable);
//	normApplyButton->setEnabled(enable || tt == TT_NORM_RANGE);
//	normClampButton->setEnabled(enable || tt == TT_CLAMP_RANGE_IMG || tt == TT_CLAMP_RANGE_GRAD);
}

void MainWindow::normTargetChanged(bool usecurrent)
{
/*
	// reset gui to current settings
	int target = (normIButton->isChecked() ? 0 : 1);
	multi_img::NormMode m = (target == 0 ? normIMG : normGRAD);

	// update normModeBox
	normModeBox->setCurrentIndex(m);

	// update norm range spin boxes
	normModeSelected(m, true, usecurrent);
	*/
}

// -> NormDock::processNormModeSelected()
void MainWindow::normModeSelected(int mode, bool targetchange, bool usecurrent)
{
	/*
	multi_img::NormMode nm = static_cast<multi_img::NormMode>(mode);
	if (nm == multi_img::NORM_FIXED && !targetchange) // user edits from currenty viewed values
		return;

	int target = (normIButton->isChecked() ? 0 : 1);

	if (!usecurrent) {
		multi_img::Value min;
		multi_img::Value max;
		if (target == 0) {
			SharedDataLock hlock(normIMGRange->mutex);
			min = (*normIMGRange)->first;
			max = (*normIMGRange)->second;
		} else {
			SharedDataLock hlock(normGRADRange->mutex);
			min = (*normGRADRange)->first;
			max = (*normGRADRange)->second;
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				nm, target, min, max, false));
			queue.push(taskNormRange);
			taskNormRange->wait();
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				nm, target, min, max, false));
			queue.push(taskNormRange);
			taskNormRange->wait();
		}
	}

	double min;
	double max;
	if (target == 0) {
		SharedDataLock hlock(normIMGRange->mutex);
		min = (*normIMGRange)->first;
		max = (*normIMGRange)->second;
	} else {
		SharedDataLock hlock(normGRADRange->mutex);
		min = (*normGRADRange)->first;
		max = (*normGRADRange)->second;
	}

	// prevent signal loop
	normMinBox->blockSignals(true);
	normMaxBox->blockSignals(true);
	normMinBox->setValue(min);
	normMaxBox->setValue(max);
	normMinBox->blockSignals(false);
	normMaxBox->blockSignals(false);
	*/
}

void MainWindow::normModeFixed()
{
//	if (normModeBox->currentIndex() != 2)
//		normModeBox->setCurrentIndex(2);
}

void MainWindow::applyNormUserRange()
{
	/*
	representation::t target = (normIButton->isChecked() ?
							representation::IMG : representation::GRAD);

	// set internal norm mode
	multi_img::NormMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<multi_img::NormMode>(normModeBox->currentIndex());

	queue.cancelTasks();
	setGUIEnabled(false, TT_NORM_RANGE);

	// if available, overwrite with more precise values than in the spin boxes.
	if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
		BackgroundTaskPtr taskNormRange(new NormRangeCuda(
			(target == 0 ? image : gradient), 
			(target == 0 ? normIMGRange : normGRADRange), 
			(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
		queue.push(taskNormRange);
	} else {
		BackgroundTaskPtr taskNormRange(new NormRangeTbb(
			(target == 0 ? image : gradient), 
			(target == 0 ? normIMGRange : normGRADRange), 
			(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
		queue.push(taskNormRange);
	}

	// re-initialize gui (duplication from applyROI())
	if (target == 0) {
//		viewIMG->updateBinning(-1);
		viewerContainer->updateBinning(representation::IMG,-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			viewerContainer, SLOT(finishNormRangeImgChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	} else {
//		viewGRAD->updateBinning(-1);
		viewerContainer->updateBinning(GRAD,-1);

		BackgroundTaskPtr taskFinishNorm(new BackgroundTask());
		QObject::connect(taskFinishNorm.get(), SIGNAL(finished(bool)), 
			viewerContainer, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishNorm);
	}

	BackgroundTaskPtr taskEpilog(new BackgroundTask());
	QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
		this, SLOT(finishTask(bool)), Qt::QueuedConnection);
	queue.push(taskEpilog);
	*/
}

// TODO missing in ImageModel
void MainWindow::clampNormUserRange()
{
	/*
	int target = (normIButton->isChecked() ? 0 : 1);

	// set internal norm mode
	multi_img::NormMode &nm = (target == 0 ? normIMG : normGRAD);
	nm = static_cast<multi_img::NormMode>(normModeBox->currentIndex());

	/// if image is changed, change full image. for gradient, we cannot preserve
	///	the gradient over ROI or illuminant changes, so it remains a local change
	if (target == 0) {
		queue.cancelTasks(roi);
		setGUIEnabled(false, TT_CLAMP_RANGE_IMG);

		// if available, overwrite with more precise values than in the spin boxes.
		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true, roi));
			queue.push(taskNormRange);
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true, roi));
			queue.push(taskNormRange);
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_CLAMP) {
			BackgroundTaskPtr taskClamp(new ClampCuda(
				image_lim, image, roi, false));
			queue.push(taskClamp);
		} else {
			BackgroundTaskPtr taskClamp(new ClampTbb(
				image_lim, image, roi, false));
			queue.push(taskClamp);
		}

		applyROI(false);
		falseColorDock->setEnabled(false);

		// create task to compute rgb image in parallel
		BackgroundTaskPtr taskRgb(new RgbTbb(
			image_lim, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), full_rgb_temp, roi));
		// connect finished signal to our update slot
		// the task will then trigger the GUI update to show computation result
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)), this, SLOT(updateRGB(bool)), Qt::QueuedConnection);
		// after signal is connected, enqueue the task in our taskqueue
		queue.push(taskRgb);

		BackgroundTaskPtr taskEpilog(new BackgroundTask(roi));
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	} else {
		queue.cancelTasks();
		setGUIEnabled(false, TT_CLAMP_RANGE_GRAD);

		// if available, overwrite with more precise values than in the spin boxes.
		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_DATARANGE) {
			BackgroundTaskPtr taskNormRange(new NormRangeCuda(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
			queue.push(taskNormRange);
		} else {
			BackgroundTaskPtr taskNormRange(new NormRangeTbb(
				(target == 0 ? image : gradient), 
				(target == 0 ? normIMGRange : normGRADRange), 
				(target == 0 ? normIMG : normGRAD), target, normMinBox->value(), normMaxBox->value(), true));
			queue.push(taskNormRange);
		}

		if (cv::gpu::getCudaEnabledDeviceCount() > 0 && USE_CUDA_CLAMP) {
			BackgroundTaskPtr taskClamp(new ClampCuda(gradient, gradient));
			queue.push(taskClamp);
		} else {
			BackgroundTaskPtr taskClamp(new ClampTbb(gradient, gradient));
			queue.push(taskClamp);
		}

//		viewGRAD->updateBinning(-1);
		viewerContainer->updateBinning(GRAD,-1);

		BackgroundTaskPtr taskFinishClamp(new BackgroundTask());
		QObject::connect(taskFinishClamp.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishNormRangeGradChange(bool)), Qt::QueuedConnection);
		queue.push(taskFinishClamp);

		BackgroundTaskPtr taskEpilog(new BackgroundTask());
		QObject::connect(taskEpilog.get(), SIGNAL(finished(bool)), 
			this, SLOT(finishTask(bool)), Qt::QueuedConnection);
		queue.push(taskEpilog);
	}
	*/
}

void MainWindow::openContextMenu()
{
	delete contextMenu;
	contextMenu = createPopupMenu();
	contextMenu->exec(QCursor::pos());
}

void MainWindow::screenshot()
{
	// grabWindow reads from the display server, so GL parts are not missing
	QPixmap shot = QPixmap::grabWindow(this->winId());

	// we use OpenCV so the user can expect the same data type support
	cv::Mat output = vole::QImage2Mat(shot.toImage());

	IOGui io("Screenshot File", "screenshot", this);
	io.writeFile(QString(), output);
}

void MainWindow::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch (e->type()) {
	case QEvent::LanguageChange:
		retranslateUi(this);
		break;
	default:
		break;
	}
}
