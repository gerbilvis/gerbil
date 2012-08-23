#ifndef EXECUTIONTHREAD_H
#define EXECUTIONTHREAD_H

#include <QThread>
#include <QMutex>
#include <QImage>
#include <QDebug>

#include "gui_command_thread.h"
#include "cv.h"

//#include "execution.h"

namespace vole {

class ExecutionThread : public QThread
{
	Q_OBJECT

signals:
	void imgStart();
	void imgDone(QImage*);

public:
	ExecutionThread();
	QImage getImg();

	void setWorker(GuiCommandThread *worker);

protected:
	void run();

private:

	QImage qtImg;
	cv::Mat* bImg;

	GuiCommandThread *worker;
};

}

#endif // EXECUTIONTHREAD_H
