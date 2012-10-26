#include "viewerwindow.h"
#include <background_task_queue.h>
#include <tbb/compat/thread>
#include <QApplication>
#include <QFileDialog>
#include <iostream>
#include <string>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

 #include <QGLFormat>
 #include <QGLFramebufferObject>

#ifdef __GNUC__
#include <tr1/functional>
#endif

/** All OpenCV functions that are called from parallelized parts of gerbil
    have to be first executed in single-threaded environment. This is actually
    required only for functions that contain 'static const' variables, but to 
    avoid investigation of OpenCV sources and to defend against any future
    changes in OpenCV, it is advised not to omit any used function. Note that
    'static const' variables within functions are initialized in a lazy manner
    and such initialization is not thread-safe because setting the value and
    init flag of the variable is not an atomic operation. */
void init_opencv()
{
	double d1, d2;
	multi_img::Band b1(1, 1);
	multi_img::Band b2(1, 1);
	multi_img::Band b3(1, 2);

	b1(0, 0) = 1.0;
	b2(0, 0) = 1.0;
	b3(0, 0) = 1.0;
	b3(0, 1) = 1.0;

	cv::minMaxLoc(b1, &d1, &d2);
	cv::resize(b3, b2, cv::Size(1, 1));
	cv::log(b1, b2);
	cv::max(b1, 0., b2);
	cv::subtract(b1, b1, b2);
	cv::multiply(b1, b1, b2);
	cv::divide(b1, b1, b2);
	cv::PCA pca(b1, cv::noArray(), CV_PCA_DATA_AS_COL, 0);
	pca.project(b1, b2);
}

void init_cuda()
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
}

#ifdef __GNUC__
#define cpuid(func, ax, bx, cx, dx)\
	__asm__ __volatile__ ("cpuid":\
	"=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#endif

bool test_compatibility()
{
	bool supportMMX = false;
	bool supportSSE = false;
	bool supportSSE2 = false;

	int info[4];
	info[0] = 0x7fffffff;
	info[1] = 0x7fffffff;
	info[2] = 0x7fffffff;
	info[3] = 0x7fffffff;

	#ifdef _MSC_VER
	__cpuid(info, 0);
	#endif

	#ifdef __GNUC__
	cpuid(0, info[0], info[1], info[2], info[3])
	#endif
	
	int nIds = info[0];

	if (nIds >= 1){
		#ifdef _MSC_VER
		__cpuid(info, 1);
		#endif

		#ifdef __GNUC__
		cpuid(1, info[0], info[1], info[2], info[3])
		#endif

		supportMMX = (info[3] & ((int)1 << 23)) != 0;
		supportSSE = (info[3] & ((int)1 << 25)) != 0;
		supportSSE2 = (info[3] & ((int)1 << 26)) != 0;
	}

	bool supportOGL = QGLFormat::hasOpenGL();
	bool supportFBO = QGLFramebufferObject::hasOpenGLFramebufferObjects();
	bool supportBlit = QGLFramebufferObject::hasOpenGLFramebufferBlit();

	if (!supportMMX) std::cout << "MMX support not found." << std::endl;
	if (!supportSSE) std::cout << "SSE support not found." << std::endl;
	if (!supportSSE2) std::cout << "SSE2 support not found." << std::endl;
	if (!supportOGL) std::cout << "OpenGL support not found." << std::endl;
	if (!supportFBO) std::cout << "GL_EXT_framebuffer_object support not found." << std::endl;
	if (!supportBlit) std::cout << "GL_EXT_framebuffer_blit support not found." << std::endl;

	bool success = supportMMX && supportSSE && supportSSE2 && supportOGL && supportFBO && supportBlit;

	if (!success) std::cout << "Machine does not meet minimal requirements to launch Gerbil." << std::endl;

	return success;
}

int main(int argc, char **argv)
{
	init_opencv();
	init_cuda();

	// start gui
	QApplication app(argc, argv);

	if (!test_compatibility())
		return -1;

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

