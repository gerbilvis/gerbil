#include <qtopencv.h>
#include "executionthread.h"
#include <iostream>

namespace vole {

ExecutionThread::ExecutionThread() {}

void ExecutionThread::run() {
	emit imgStart();

//	bImg = worker->execute_headless();
	int status = worker->executeThread();
	std::cout << "status: " << status << std::endl;

	// FIXME fill bImg
	cv::Mat_<cv::Vec3b> output(*bImg);
	qtImg = Mat2QImage(output);

	emit imgDone(&qtImg);
}

void ExecutionThread::setWorker(GuiCommandThread *work) {
	worker = work;
}

QImage ExecutionThread::getImg() {
	return qtImg;
}

}
