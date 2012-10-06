#include "viewerwindow.h"
#include <background_task_queue.h>
#include <tbb/compat/thread>
#include <QApplication>
#include <QFileDialog>
#include <iostream>
#include <string>

#include <opencv2/gpu/gpu.hpp>

#ifdef __GNUC__
#include <tr1/functional>
#endif

int main(int argc, char **argv)
{
	if (cv::gpu::getCudaEnabledDeviceCount() > 0) {
		cv::gpu::DeviceInfo info;

		std::cout << "Initializing CUDA..." << std::endl;
		info.totalMemory(); // trigger CUDA initialization (just-in-time compilation etc.)
		std::cout << std::endl;

		std::cout << "Found CUDA compatible device: " << std::endl;
		std::cout << "Device ID: " << info.deviceID() << std::endl;
		std::cout << "Device name: " << info.name() << std::endl;
		std::cout << "Multiprocessor count: " << info.multiProcessorCount() << std::endl;
		std::cout << "Free memory: " << info.freeMemory() << std::endl;
		std::cout << "Total memory: " << info.totalMemory() << std::endl;
		std::cout << "Compute capability: " << info.majorVersion() << "." << info.minorVersion() << std::endl;
		std::cout << "Global atomics support: " << info.supports(cv::gpu::GLOBAL_ATOMICS) << std::endl;
		std::cout << "Shared atomics support: " << info.supports(cv::gpu::SHARED_ATOMICS) << std::endl;
		std::cout << "Native double support: " << info.supports(cv::gpu::NATIVE_DOUBLE) << std::endl;
		std::cout << std::endl;

		std::cout << "OpenCV GPU module information: " << std::endl;
		std::cout << "Compute capability 1.0 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(1, 0) << ":" << 
			cv::gpu::TargetArchs::hasBin(1, 0) << std::endl;
		std::cout << "Compute capability 1.1 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(1, 1) << ":" << 
			cv::gpu::TargetArchs::hasBin(1, 1) << std::endl;
		std::cout << "Compute capability 1.2 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(1, 2) << ":" << 
			cv::gpu::TargetArchs::hasBin(1, 2) << std::endl;
		std::cout << "Compute capability 1.3 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(1, 3) << ":" << 
			cv::gpu::TargetArchs::hasBin(1, 3) << std::endl;
		std::cout << "Compute capability 2.0 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(2, 0) << ":" << 
			cv::gpu::TargetArchs::hasBin(2, 0) << std::endl;
		std::cout << "Compute capability 2.1 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(2, 1) << ":" << 
			cv::gpu::TargetArchs::hasBin(2, 1) << std::endl;
		std::cout << "Compute capability 3.0 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(3, 0) << ":" << 
			cv::gpu::TargetArchs::hasBin(3, 0) << std::endl;
		std::cout << "Compute capability 3.5 [PTX:BIN]: " <<  
			cv::gpu::TargetArchs::hasPtx(3, 5) << ":" << 
			cv::gpu::TargetArchs::hasBin(3, 5) << std::endl;
		std::cout << "Global atomics support: " << cv::gpu::TargetArchs::builtWith(cv::gpu::GLOBAL_ATOMICS) << std::endl;
		std::cout << "Shared atomics support: " << cv::gpu::TargetArchs::builtWith(cv::gpu::SHARED_ATOMICS) << std::endl;
		std::cout << "Native double support: " << cv::gpu::TargetArchs::builtWith(cv::gpu::NATIVE_DOUBLE) << std::endl;
		std::cout << std::endl;
	}

	// start gui
	QApplication app(argc, argv);

	// start worker thread
	std::thread background(std::tr1::ref(BackgroundTaskQueue::instance()));

	// get input file name
	std::string filename;
	if (argc < 2) {
#ifdef __unix__
		std::cerr << "Usage: " << argv[0] << " <filename> [labeling file]\n\n"
					 "Filename may point to a RGB image or "
					 "a multispectral image descriptor file." << std::endl;
#endif
		filename = QFileDialog::getOpenFileName
		           	(0, "Open Descriptor or Image File").toStdString();
	} else {
		filename = argv[1];
	}

	QString labelfile;
	if (argc >= 3)
		labelfile = argv[2];

	// load image   
	multi_img* image = new multi_img(filename);
	if (image->empty())
		return 2;
	
	// regular viewer
	ViewerWindow window(image);
	image = NULL;
	window.show();
	
	// load labels
	if (!labelfile.isEmpty())
		window.loadLabeling(labelfile);

	int retval = app.exec();

	// terminate worker thread
	BackgroundTaskQueue::instance().halt();

	// wait until worker thread terminates
	background.join();
	
	return retval;
}

