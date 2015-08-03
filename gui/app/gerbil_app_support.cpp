
#include "gerbil_app_support.h"
#include <gerbilapplication.h>
#include <dialogs/openrecent/recentfile.h>

#ifdef GERBIL_CUDA
	#include <opencv2/gpu/gpu.hpp>
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QMessageBox>
#include <QGLFormat>
#include <QGLFramebufferObject>
#include <QMessageBox>
//#include <QFileDialog>
#include <QPushButton>

#ifdef __GNUC__
#define cpuid(func, ax, bx, cx, dx)\
	__asm__ __volatile__ ("cpuid":\
	"=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#endif

#ifdef GERBIL_CUDA
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
#endif

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


/** Determines just a rough estimated range of memory requirements to accomodate
	input data for Gerbil startup. Data structures whose size do not depend on
	input are not accounted for (framebuffers, greyscale thumbnails, etc.).
	Overhead of data structures and heap allocator is also not accounted for. */
void estimate_startup_memory(int width, int height, int bands,
	float &lo_reg, float &hi_reg, float &lo_opt, float &hi_opt, float &lo_gpu, float &hi_gpu)
{
	// full multi_img, assuming no pixel cache
	float full_img = width * height * bands * sizeof(multi_img::Value) / 1048576.;
	// full RGB image, assuming ARGB format
	float rgb_img = width * height * 4 / 1048576.;
	// labeling matrix
	float lab_mat = width * height * sizeof(short) / 1048576.;
	// scoped multi_img, assuming ROI and pixel cache
	float scoped_img = ((width > 512) ? 512 : width) * ((height > 512) ? 512 : height)
		* bands * sizeof(multi_img::Value) * 2 / 1048576.;
	// hash table and shuffling vector for extremely noisy data
	float hashing_max = ((width > 512) ? 512 : width) * ((height > 512) ? 512 : height)
		* bands * sizeof(multi_img::Value) * 2 / 1048576.;
	// vertex buffer for extremely noisy data
	float vbo_max = ((width > 512) ? 512 : width) * ((height > 512) ? 512 : height)
		* bands * 2 * sizeof(float) / 1048576.;

	// data without too much noise, hashing yields significant savings with default bin count
	lo_reg = full_img + (2 * scoped_img) + rgb_img + lab_mat + (2 * hashing_max * 0.15);
	lo_opt = (2 * scoped_img) + rgb_img + lab_mat + (2 * hashing_max * 0.15);
	lo_gpu = rgb_img + (2 * vbo_max) * 0.15;

	// noisy data, hashing is not very effective
	hi_reg = full_img + (2 * scoped_img) + rgb_img + lab_mat + (2 * hashing_max * 0.8);
	hi_opt = (2 * scoped_img) + rgb_img + lab_mat + (2 * hashing_max * 0.8);
	hi_gpu = rgb_img + (2 * vbo_max) * 0.8;
}



bool determine_limited(const std::pair<std::vector<std::string>,
					   std::vector<multi_img::BandDesc> > &filelist)
{
	if (!filelist.first.empty()) {
		cv::Mat src = cv::imread(filelist.first[1], -1);
		if (!src.empty()) {
			float lo_reg, hi_reg, lo_opt, hi_opt, lo_gpu, hi_gpu;
			estimate_startup_memory(src.cols, src.rows, src.channels() * filelist.first.size(),
				lo_reg, hi_reg, lo_opt, hi_opt, lo_gpu, hi_gpu);

			// default speed optim. in case of smaller images
			if (hi_reg < 512)
				return false;

			/* TODO: move. it does not work here because GL context is missing.
			GLint maxTextureSize;
			glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);
			std::cout << maxTextureSize << std::endl;
			if (src.cols * src.rows > maxTextureSize) {
				std::cout << "WARNING: Graphics device does not support texture size "
					<< "required to render RGB version of input image in full resolution. " << std::endl;
			}
			 */

			std::stringstream text;
			text << "Please consider memory requirements:"
					"<ul>"
					"<li>Speed optim.: <b>" << (int)lo_reg << "</b> to <b>"
											<< (int)hi_reg << "</b> MB"
					"<li>Space optim.: <b>" << (int)lo_opt << "</b> to <b>"
											<< (int)hi_opt << "</b> MB"
					"<li>GPU memory:   <b>" << (int)lo_gpu << "</b> to <b>"
											<< (int)hi_gpu << "</b> MB"
					"</ul>"
					"Please choose between speed and space optimization or close "
					"the program in case of insufficient system ressources.";

	/*			text << "For startup, Gerbil will have to allocate between "
				<< lo_reg << "MB and " << hi_reg
				<< "MB of memory to accommodate data derived from input image. "
				<< "At performance cost and some disabled features, "
				<< "memory consumption can be optimized to range between "
				<< lo_opt << "MB and " << hi_opt << "MB. "
				<< "Additionaly, between "
				<< lo_gpu << "MB and " << hi_gpu << "MB of GPU memory will be required. "
				<< "Note that estimated requirements do not include Gerbil itself "
				<< "and overhead of its storage mechanisms. Depending on the characteristics "
				<< "of your machine (CPU/GPU RAM size, page file size, HDD/SSD performance), "
				<< "decide whether to optimize performance or memory consumption. You can also "
				<< "close Gerbil to avoid possible memory exhaustion and computer lock-up. ";
	*/
			QMessageBox msgBox;
			msgBox.setText(text.str().c_str());
			msgBox.setIcon(QMessageBox::Question);
			QPushButton *speed = msgBox.addButton("Speed optimization", QMessageBox::AcceptRole);
			QPushButton *memory = msgBox.addButton("Memory optimization", QMessageBox::AcceptRole);
			QPushButton *close = msgBox.addButton("Close", QMessageBox::RejectRole);
			msgBox.setDefaultButton(speed);
			msgBox.exec();
			if (msgBox.clickedButton() == memory)
				return true;
			if (msgBox.clickedButton() == close)
				exit(GerbilApplication::ExitSuccess);
		}
	}

	// if we could not read the image this way, default to no limited mode
	return false;
}

void registerQMetaTypes()
{
	qRegisterMetaType<RecentFile>("RecentFile");
	qRegisterMetaTypeStreamOperators<RecentFile>("RecentFile");
}


bool check_system_requirements()
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

	// FIXME: error window
	if (!supportMMX) std::cerr << "MMX support not found." << std::endl;
	if (!supportSSE) std::cerr << "SSE support not found." << std::endl;
	if (!supportSSE2) std::cerr << "SSE2 support not found." << std::endl;
	if (!supportOGL) std::cerr << "OpenGL support not found." << std::endl;
	if (!supportFBO) std::cerr << "GL_EXT_framebuffer_object support not found." << std::endl;
	if (!supportBlit) std::cerr << "GL_EXT_framebuffer_blit support not found." << std::endl;

	bool success = supportMMX && supportSSE && supportSSE2 && supportOGL && supportFBO && supportBlit;
	return success;
}
