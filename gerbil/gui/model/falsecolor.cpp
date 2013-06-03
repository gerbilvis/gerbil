
#include "falsecolor.h"

#include <background_task_queue.h>
#include <multi_img.h>
#include <multi_img_tasks.h>
#include <qtopencv.h>
#include <rgb.h>
#include <shared_data.h>

#include <tasks/rgbtbb.h> // TODO: mit pfad?

#include <QImage>
#include <opencv2/core/core.hpp>

// TODO:
// reset() -> empty job queue - wie?
// was passiert mit finished-Signalen, die schon abgeschickt, aber noch nicht bearbeitet wurden?
// was passiert, wenn reset aus setMultiImg aufgerufen wird, aber bilder bereits angefragt sind?
//  \-> momentan wird der runner abgebrochen, das widget bekommt davon aber nichts mit und wartet ewig
//  \-> cancelled signal oder sowas? dann kann das widget neu anfragen, falls das bild noch benoetigt wird...

// commandrunner.cpp
//  \-> why is abort not synchronized in any way?

// RGB:
//  \-> RGB hat Boost momentan als optional dependency, vole braucht sie nicht, wenn man die
//      Funktion mit boost parameter verwenden kann, dann hat man auch Boost
//      Soll das wirklich optional bleiben?
//  \-> progressUpdate() -> PCA Werte anpassen, mehr updates?, SOM einbauen,
//		XYZ verwendet multi_img methode und kann somit kein update aufrufen
//  \-> Im output speichern, welcher algorithmus berechnet wurde, ist nicht gerade toll...
//       \-> fuers RGB modul wuerde es besser passen, wenn model auch mit gerbil::rgbalg arbeiten wuerde...
//  \-> wie krieg ich den header zum compilieren, eig mit class ProgressObserver, namespace?

// Wie funktioniert der Swap Lock genau, siehe SharedDataSwapLock Versuch ganz unten
// Alternative: QSignalMapper, passt gut zum enum, der kann aber nur Signale ohne Parameter
//  \-> Wenn das Ergebnis im Parameter uebergeben wird geht's nicht
//  \-> Evtl die output map im CommandRunner speichern?


// Long term TODOs:
// "init rgb.configs, if non default setup is neccessary"

// sichergehen, dass img immer der aktuelle ROI ausschnitt ist
// ROI per signal slot verteilen <-> code georg --> sollte dann hier egal sein, weil das img ja immer passend gesetzt werden sollte

// const multi_img* statt SharedMultiImg
//  \-> __Momentan obsolete, da SharedPointer verwendet wird__
//      Auf Img (Konstruktor) wird ein Pointer gespeichert.
//      Dieses (Img) muss bestehen bleiben, bis
//      a) setMultiImg mit einem anderen Img ausgefuehrt wird
//      b) das FalseColor object deleted wird

// Langfristige Frage: Sind QImages oder QPixmaps interessant? (oder beides), was davon soll nur 1x im model sein?
//  \-> "QPixmaps cannot be directly shared between threads" -> model sollte aber im *einen* GUI-Thread sein.

FalseColor::FalseColor(SharedMultiImgPtr shared_img, BackgroundTaskQueue queue)
	: shared_img(shared_img), queue(queue)
//FalseColor::FalseColor(const multi_img &img, BackgroundTaskQueue queue) : img(&img), queue(queue)
{
	for (int i = 0; i < COLSIZE; ++i) {
#ifndef WITH_EDGE_DETECT
		if (i == SOM)
			continue;
#endif
		payload *p = new payload;
		map.insert((coloring)i, p);
	}

	reset();
}

void FalseColor::setMultiImg(SharedMultiImgPtr shared_img)
//void FalseColor::setMultiImg(const multi_img *img)
{
	this->shared_img = shared_img;
	//this->img = img;

	reset();
}

FalseColor::~FalseColor()
{
	terminateTasksDeleteRunners();
}

void FalseColor::reset()
{
	terminateTasksDeleteRunners();

	// reset all images
	PayloadList l = map.values();
	foreach(payload *p, l) {
		p->img = QImage();
		p->calcImg = qimage_ptr(new SharedData<QImage>(new QImage()));
		p->calcInProgress = false;
		/* TODO: maybe send the empty image as signal to
		 * disable viewing of obsolete information
		 * maybe add bool flag, so this doesn't happen at construction
		 */
	}
}

void FalseColor::terminateTasksDeleteRunners()
{
	emit terminateRunners();

	// TODO: empty job queue & wait until it's done, else we could get a finished-Signal after resetting

	// wait & destroy all runners
	PayloadList l = map.values();
	foreach(payload *p, l) {
		if (p->runner != NULL) {
			p->runner->wait();
			delete p->runner;
			p->runner = NULL;
		}
	}
}

void FalseColor::createRunner(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);
	assert(p->runner == NULL); // runners are only deleted in reset(). if reset() was not called and the
	// runner exists, the image has been calculated already, so there is no reason to call craeteRunner

	// init runner & command
	p->runner = new CommandRunner();	// TODO: hier wuerde ein parent nicht schaden...
	p->runner->cmd = new gerbil::RGB(); // the RGB object is deleted in ~CommandRunner()

	// TODO: init rgb.config, if non-default setup is neccessary. then we need a copy of
	// the initialized RGB object, as the obj in the runner is deleted in its destructor
	gerbil::RGB *cmd = (gerbil::RGB *)p->runner->cmd;
	switch (type)
	{
	case CMF:
		cmd->config.algo = gerbil::COLOR_XYZ;
		break;
	case PCA:
		cmd->config.algo = gerbil::COLOR_PCA;
		break;
#ifndef WITH_EDGE_DETECT
	case SOM:
		cmd->config.algo = gerbil::COLOR_SOM;
		break;
#endif
	default:
		assert(false); // coloring type is COLSIZE, SOM without edge detect or missing in the switch
	}

	QObject::connect(this, SIGNAL(terminateRunners()),
					 p->runner, SLOT(terminate()), Qt::QueuedConnection);
	QObject::connect(p->runner, SIGNAL(success(std::map<std::string, boost::any>)),
					 this, SLOT(handleRunnerSuccess(std::map<std::string, boost::any>)), Qt::QueuedConnection);
}


void FalseColor::calculateForeground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit loadComplete(p->img, type);
		return;
	}

	// img is currently in calculation, loadComplete will be emitted as soon as its finished
	if (p->calcInProgress)
		return;

	// we can't get around doing some real calculations
	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), p->calcImg));
		taskRgb.get()->run();
	}
	else {
		createRunner(type);
		cv::Mat3b mat = (cv::Mat3b)(((gerbil::RGB *)p->runner->cmd)->execute(**shared_img) * 255.0f);
		//cv::Mat3b mat = (cv::Mat3b)(((gerbil::RGB *)p->runner->cmd)->execute(*img) * 255.0f);
		p->img = vole::Mat2QImage(mat);
	}
	p->calcInProgress = false;
	emit loadComplete(p->img, type);
}

void FalseColor::calculateBackground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	// img is calculated already
	if (!p->img.isNull()) {
		emit loadComplete(p->img, type);
		return;
	}

	// img is currently in calculation, loadComplete will be emitted as soon as its finished
	if (p->calcInProgress)
		return;

	// we can't get around doing some real calculations
	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), p->calcImg));
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)),
						 this, SLOT(handleFinishedQueueTask(bool)), Qt::QueuedConnection);
		queue.push(taskRgb);
	}
	else {
		createRunner(type);
		p->runner->start();
	}
}


// Only CMF tasks are calculated in the queue
void FalseColor::handleFinishedQueueTask(bool success)
{
	coloring type = CMF;
	payload *p = map.value(type);
	assert(p != NULL);

	// TODO: wenn !success evtl wieder anstossen... eher nicht, das sollte eig nur passieren, wenn reset() aufgerufen wird...

	// TODO: vom "tmp"-member (calcImg) in eigtl. var (img) "kopieren"
	{
		SharedDataSwapLock lock(p->calcImg->mutex);
		p->img = **p->calcImg;
		delete p->calcImg->swap(new QImage); // TODO: locks, empty the reference - how is it done correctly?
	}

	p->calcInProgress = false;
	emit loadComplete(p->img, type);
}

void FalseColor::handleRunnerSuccess(std::map<std::string, boost::any> output)
{
	// TODO: This will lead to problems, if we need to separate between e.g. img_pca and grad_pca,
	// because RGB cannot (and should not be able to) differentiate between the two...
	coloring type;
	switch (boost::any_cast<gerbil::rgbalg>(output["algo"])) {
	case gerbil::COLOR_XYZ:
		type = CMF;
		break;
	case gerbil::COLOR_PCA:
		type = PCA;
		break;
	case gerbil::COLOR_SOM:
		type = SOM;
		break;
	default:
		assert(false); // wrong output["algo"] value
	}

	payload *p = map.value(type);
	p->img = boost::any_cast<QImage>(output["multi_img"]);

	p->calcInProgress = false;
	emit loadComplete(p->img, type);
}
