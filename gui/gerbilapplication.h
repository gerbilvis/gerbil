#ifndef GERBILAPPLICATION_H
#define GERBILAPPLICATION_H

#include <QApplication>
#include <cstdlib>

#include <gerbil_cplusplus.h>

class GerbilApplication : public QApplication
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

		// No input file given.
		ExitNoInput =				EXIT_FAILURE + 3
	};

	explicit GerbilApplication ( int & argc, char ** argv );
	explicit GerbilApplication ( int & argc, char ** argv, bool GUIenabled );

	/** Execute the gerbil GUI application.
	 *
	 * This checks various pre-conditions for running, sets up objects and
	 * executes the Qt event loop (QApplication::exec()).
	 * This functions does not return but calls exit().
	 * @see ExitStatus.
	 */
	// TODO make this nothrow and catch all exceptions here.
	virtual void run();
signals:

public slots:

private:

};

#endif // GERBILAPPLICATION_H
