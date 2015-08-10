#ifndef GERBILAPPLICATION_H
#define GERBILAPPLICATION_H

#include <multi_img.h>

#include <boost/noncopyable.hpp>

#include <QApplication>
#include <exception>

class Controller;

class GerbilApplication : public QApplication, boost::noncopyable
{
	Q_OBJECT
public:

	explicit GerbilApplication(int & argc, char ** argv);

	/** Returns a pointer to the GerbilApplication object. */
	static GerbilApplication *instance()
	{
		return dynamic_cast<GerbilApplication*>(QCoreApplication::instance());
	}

	/** Returns the path to the directory to the loaded multi-spectral
	 * image file without the filename.
	 *
	 * If no image is loaded this returns a null string.
	 */
	QString imagePath();

public slots:
	/** Execute the gerbil GUI application.
	 *
	 * This checks various pre-conditions for running and sets up objects
	 * It expects the Qt event loop to be up.
	 */
	// TODO make this nothrow and catch all exceptions here.
	virtual void run();

	/** Display a critical error in a message box and kill the application */
	void userError(QString msg);
	void internalError(QString msg, bool critical = true);
	void criticalError(QString msg, bool quit = true);

	/** Print error on stderr and in message box (at least tries to).
	 * If critical is set, also quit the application.
	 */
	void handle_exception(std::exception_ptr e, bool critical);

protected:
	/**  Initialize OpenCV state.
	 *
	 * All OpenCV functions that are called from parallelized parts of gerbil have
	 * to be first executed in single-threaded environment. This is actually
	 * required only for functions that contain 'static const' variables, but to
	 * avoid investigation of OpenCV sources and to defend against any future
	 * changes in OpenCV, it is advised not to omit any used function. Note that
	 * 'static const' variables within functions are initialized in a lazy manner
	 * and such initialization is not thread-safe because setting the value and
	 * init flag of the variable is not an atomic operation.
	 */
	void init_opencv();

	/** Initialize CUDA. */
	void init_cuda();

	/** setup resources, icon theme, register types with Qt type system. */
	void init_qt();

	/** Check for system requirements.
	 *
	 * Checks for MXX, SSE and SSE2 and for required OpenGL features.
	 *
	 * @return Returns true if all requirements are met, otherwise false.
	 */
	void check_system_requirements();

	/** Tests wether or not to load the multi_img in limited mode.
	 *
	 * Opens dialog for querying the user. Calls exit() if user decides to close
	 * the application.
	 *
	 * @return true if multi_img should be loaded in limited mode, otherwise false.
	 */
	// FIXME use QString for filenames (full unicode support).
	bool determine_limited(const std::pair<std::vector<std::string>,
								  std::vector<multi_img::BandDesc> > &filelist);

	/** Parse arguments and open dialog in lack thereof.
	 *
	 * Test for command line argument or open recent file dialog.
	 * Checks if image should be loaded in limited mode.
	 * Also checks for label file argument.
	 */
	void parse_args();

	void printUsage();

	/** True if multi-spectral image should be loaded using limited mode. */
	bool limitedMode;

	/** The input filename of the multi-spectral image. */
	QString imageFilename;

	/** The input filename of the labels. */
	QString labelsFilename;

	// The Controller.
	Controller* ctrl;
};

#endif // GERBILAPPLICATION_H
