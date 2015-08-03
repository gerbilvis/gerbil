#ifndef GERBILAPPLICATION_H
#define GERBILAPPLICATION_H

#include <cstdlib>
#include <boost/noncopyable.hpp>

#include <QApplication>

class Controller;

class GerbilApplication : public QApplication, boost::noncopyable
{
	Q_OBJECT
public:

	/** Gerbil process exit status. */
	enum ExitStatus {
		ExitSuccess =				EXIT_SUCCESS,

		// General Failure.
		ExitFailure =				EXIT_FAILURE,

		// Minimum system requirements for Gerbil are not met.
		ExitSystemRequirements =	EXIT_FAILURE + 1,

		//                          EXIT_FAILURE + 2

		// No input file given.
		ExitNoInput =				EXIT_FAILURE + 3
	};

	explicit GerbilApplication(int & argc, char ** argv);

	/** Returns a pointer to the GerbilApplication object.
	 *
	 * Analogous to QCoreApplication::instance(), this returns a null
	 * pointer if no GerbilApplication instance has been allocated.
	 */
	static GerbilApplication *instance();

	/** Execute the gerbil GUI application.
	 *
	 * This checks various pre-conditions for running, sets up objects and
	 * executes the Qt event loop (QApplication::exec()).
	 * This functions does not return but calls exit().
	 * @see ExitStatus.
	 */
	// TODO make this nothrow and catch all exceptions here.
	virtual void run();

	/** Returns the path to the directory to the loaded multi-spectral
	 * image file without the filename.
	 *
	 * If no image is loaded this returns a null string.
	 */
	QString imagePath();

	/** Display a critical error in a message box and exit the application
	 * with an error state. */
	void userError(QString msg);
	void internalError(QString msg);
	void criticalError(QString msg);

	// for eventLoopStarted
	bool eventFilter(QObject *obj, QEvent *event) override;

private:

	/** Load multi-spectral image data.
	 *
	 * Load multi-spectral image given as command line argument or open
	 * recent file dialog. Checks if image should be loaded in limited
	 * mode.
	 */
	void loadInput();

	void printUsage();

	/** True if multi-spectral image should be loaded using limited mode. */
	bool limitedMode;

	/** The input filename of the multi-spectral image. */
	QString imageFilename;

	/** The input filename of the labels. */
	QString labelsFilename;

	// Track start of the Qt main event-loop
	int  eventLoopStartedEvent;
	volatile bool eventLoopStarted;

	// The Controller.
	Controller* ctrl;
};

#endif // GERBILAPPLICATION_H
