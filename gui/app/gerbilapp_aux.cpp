
#include "gerbilapplication.h"
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

#include <iostream>

#ifdef __GNUC__
#define cpuid(func, ax, bx, cx, dx)\
	__asm__ __volatile__ ("cpuid":\
	"=a" (ax), "=b" (bx), "=c" (cx), "=d" (dx) : "a" (func));
#endif

void GerbilApplication::check_system_requirements()
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

	QString problems;
	if (!supportMMX)
		problems += "MMX support not found.<br/>";
	if (!supportSSE)
		problems += "SSE support not found.<br/>";
	if (!supportSSE2)
		problems += "SSE2 support not found.<br/>";
	if (!supportOGL)
		problems += "OpenGL support not found.<br/>";
	if (!supportFBO)
		problems += "GL_EXT_framebuffer_object support not found.<br/>";
	if (!supportBlit)
		problems += "GL_EXT_framebuffer_blit support not found.<br/>";

	if (problems.count() > 0)
		userError(problems);
}

void GerbilApplication::init_qt()
{
	/* qt docs:
	 * "If you have resources in a static library, you might need to force
	 *  initialization of your resources"
	 */
	Q_INIT_RESOURCE(gerbil);

	// setup our custom icon theme if there is no system theme (OS X, Windows)
	if (QIcon::themeName().isEmpty() || !QIcon::themeName().compare("hicolor"))
		QIcon::setThemeName("Gerbil");

	qRegisterMetaType<std::exception_ptr>("std::exception_ptr");
	qRegisterMetaType<RecentFile>("RecentFile");
	qRegisterMetaTypeStreamOperators<RecentFile>("RecentFile");
}

void GerbilApplication::init_opencv()
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

#ifdef GERBIL_CUDA
void GerbilApplication::init_cuda()
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

/** Determines just a rough estimated range of memory requirements to accomodate
	input data for Gerbil startup. Data structures whose size do not depend on
	input are not accounted for (framebuffers, greyscale thumbnails, etc.).
	Overhead of data structures and heap allocator is also not accounted for. */
void estimate_startup_memory(int width, int height, int bands,
                             float &lo_reg, float &hi_reg,
                             float &lo_opt, float &hi_opt,
                             float &lo_gpu, float &hi_gpu)
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

bool GerbilApplication::determine_limited(const
                                          std::pair<std::vector<std::string>,
                                          std::vector<multi_img::BandDesc> >
                                          &filelist)
{
	if (!filelist.first.empty()) {
		cv::Mat src = cv::imread(filelist.first[1], -1);
		if (!src.empty()) {
			float lo_reg, hi_reg, lo_opt, hi_opt, lo_gpu, hi_gpu;
			estimate_startup_memory(src.cols, src.rows,
			                        src.channels() * filelist.first.size(),
			                        lo_reg, hi_reg, lo_opt, hi_opt,
			                        lo_gpu, hi_gpu);

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

			QMessageBox msgBox;
			msgBox.setText(text.str().c_str());
			msgBox.setIcon(QMessageBox::Question);
			QPushButton *speed = msgBox.addButton("Speed optimization",
			                                      QMessageBox::AcceptRole);
			QPushButton *memory = msgBox.addButton("Memory optimization",
			                                       QMessageBox::AcceptRole);
			QPushButton *close = msgBox.addButton("Close",
			                                      QMessageBox::RejectRole);
			msgBox.setDefaultButton(speed);
			msgBox.exec();
			if (msgBox.clickedButton() == memory)
				return true;
			if (msgBox.clickedButton() == close)
				quit();
		}
	}

	// if we could not read the image this way, default to no limited mode
	return false;
}
