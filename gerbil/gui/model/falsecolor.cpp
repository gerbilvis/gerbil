
#include "falsecolor.h"

#include <background_task_queue.h>
#include <multi_img.h>
#include <qtopencv.h>
#include <rgb.h>
#include <shared_data.h>
#include <tasks/rgbtbb.h> // TODO: mit pfad?

#include <QImage>
#include <opencv2/core/core.hpp>

// TODO:
// CommandRunner ruft map execute(map) auf, gibt diese map im signal zurueck
//    - gerbil::RGB hat die schnittstelle nicht
// QSignalMapper kann nur Signale ohne Parameter -> Wenn das Ergebnis im Parameter
//    uebergeben wird geht's nicht
// im output speichern, was berechnet wurde? Momentan existiert die Methode nicht
//    & man kommt nicht an die Daten -> fuer unterste Methode
// Wie funktioniert der Swap Lock genau, siehe SharedDataSwapLock Versuch ganz unten


// alles was nicht per rgbTbb queue gemacht wird, per CommandRunner, beispiel in meanshift.cpp & meanshift_shell.cpp

// Datenaustausch ThreadQueue <-> Model:
// im finished signal muesste uebergeben werden, welcher coloring type fertig ist
// mehrere funktionen geht auch nicht, sonst bringt die map nichts mehr
// weiteren bool flag im struct, der anzeigt, dass die berechnung fertig ist?
// bei jedem finished werden alle fertigen behandelt und entsprechend emits ausgeloest
// PROBLEM: dann muesste der Runner aber zumindest den bool-flag kennen, um ihn zu aendern!
// ALTERNATIVE: QSignalMapper Bsp. im viewercontainer.cpp, unterscheidung ueber index und somit direktzuordnung
// PROBLEM: das geht nur ohne Parameter in den Signalen

// sichergehen, dass img immer der aktuelle ROI ausschnitt ist
// ROI per signal slot verteilen <-> code georg --> sollte dann hier egal sein, weil das img ja immer passend gesetzt werden sollte

// __Momentan obsolete, da SharedPointer verwendet wird__
// Auf Img (Konstruktor) wird ein Pointer gespeichert
// -> dieses (Img) muss bestehen bleiben, bis
// a) setMultiImg mit einem anderen Img ausgefuehrt wird
// b) das FalseColor object deleted wird

// Jeder Request loest ein signal an alle widgets aus, egal ob sich das bild geaendert hat oder nicht
// Da nichts kopiert werden sollte, evtl nicht so schlimm. -> Boolean Variable "Changed"?
// Die Pixmaps werden aber wahrscheinlich neu erzeugt
// Langfristige Frage: Sind QImages oder QPixmaps interessant? (oder beides), was davon soll nur 1x im model sein?
//  \-> "QPixmaps cannot be directly shared between threads" ?!


// QImages have implicit data sharing, so the returned objects act as a pointer, the data is not copied
// If the QImage in the Model is changed, this QImage is (somewhat unnecessarily) copied to a new image
// This will be neccessary for background calculation anyways...

FalseColor::FalseColor(SharedMultiImgPtr shared_img, BackgroundTaskQueue queue) : shared_img(shared_img), queue(queue)
//FalseColor::FalseColor(const multi_img &img, BackgroundTaskQueue queue) : img(&img), queue(queue)
{
	for (int i = 0; i < COLSIZE; ++i) {
#ifndef WITH_EDGE_DETECT
		if (i == SOM)
			continue;
#endif
		payload *p = new payload;

		// init runner & command, connect signals
		p->runner = new CommandRunner();	// TODO: hier wuerde ein parent nicht schaden...
		p->runner->cmd = new gerbil::RGB(); // ( deleted in ~CommandRunner() )
		p->cmd = (gerbil::RGB *)p->runner->cmd;
		// TODO: connect
		QObject::connect(p->runner, SIGNAL(success(std::map<std::string, boost::any>)),
						 this, SLOT(runnerSuccess(std::map<std::string,boost::any>)), Qt::QueuedConnection);

		map.insert((coloring)i, p);
	}

	// TODO: init rgb.config, if non-default setup is neccessary
	map.value(CMF)->cmd->config.algo = gerbil::COLOR_XYZ;
	map.value(PCA)->cmd->config.algo = gerbil::COLOR_PCA;
#ifdef WITH_EDGE_DETECT
	map.value(SOM)->cmd->config.algo = gerbil::COLOR_SOM;
#endif

	resetCaches();
}

FalseColor::~FalseColor()
{
	PayloadList l = map.values();
	foreach(payload *p, l) {
		delete p->runner;
		delete p;
	}
}


void FalseColor::resetCaches()
{
	// empty job queue

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

/*void FalseColor::setMultiImg(const multi_img& img)
{
	this->img = &img;

	resetCaches();
}*/
void FalseColor::setMultiImg(SharedMultiImgPtr shared_img)
{
	this->shared_img = shared_img;

	resetCaches();
}

void FalseColor::requestForeground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	if (!p->img.isNull()) {
		emit loadComplete(p->img, type, false);
		return;
	}

	if (p->calcInProgress)
		return;

	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), p->calcImg));
		taskRgb.get()->run();
	}
	else {
		cv::Mat3b mat = (cv::Mat3b)(p->cmd->execute(**shared_img) * 255.0f);
		//cv::Mat3b mat = (cv::Mat3b)(p->cmd->execute(*img) * 255.0f);
		p->img = vole::Mat2QImage(mat);
	}
	emit loadComplete(p->img, type, true);
}

void FalseColor::requestBackground(coloring type)
{
	payload *p = map.value(type);
	assert(p != NULL);

	if (!p->img.isNull()) {
		emit loadComplete(p->img, type, false);
		return;
	}

	if (p->calcInProgress)
		return;

	p->calcInProgress = true;
	if (type == CMF) {
		BackgroundTaskPtr taskRgb(new RgbTbb(
			shared_img, mat3f_ptr(new SharedData<cv::Mat3f>(new cv::Mat3f)), p->calcImg));
		QObject::connect(taskRgb.get(), SIGNAL(finished(bool)),
						 this, SLOT(queueTaskFinished(bool)), Qt::QueuedConnection);
		queue.push(taskRgb);
	}
	else {
		p->runner->start();
	}
}

// Only CMF tasks are calculated in the queue
void FalseColor::queueTaskFinished(bool success)
{
	coloring type = CMF;
	payload *p = map.value(type);
	if (p->calcInProgress) // TODO: muss da die sichtbarkeit synchronisiert werden?
	{
		p->calcInProgress = false;

		// TODO: wenn !success evtl wieder anstossen - momentan ist aber nicht klar, welcher fehlgeschlagen ist!

		// TODO: vom "tmp"-static member in eigtl. var kopieren
		//       vlt besser: SharedData Objekt speichern, da swapt der Thread eh direkt rein
		{
			SharedDataSwapLock lock(p->calcImg->mutex);
			p->img = **p->calcImg;
			delete p->calcImg->swap(new QImage); // TODO: i have no idea what i'm doing... locks & empty the reference!
		}

		emit loadComplete(p->img, type, true);
	}
}

void FalseColor::runnerSuccess(std::map<std::string, boost::any> output)
{
	// p->img = output["todo"];
	// emit loadComplete(p->img, type, true);
}
