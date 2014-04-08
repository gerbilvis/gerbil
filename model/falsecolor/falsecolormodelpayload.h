#ifndef FALSECOLORMODELPAYLOAD_H
#define FALSECOLORMODELPAYLOAD_H


#include <cassert>
#include <map>
#include <QObject>
#include <QPixmap>

#include <boost/any.hpp>

#include "shared_data.h"
#include "../representation.h"
#include "falsecoloring.h"

class CommandRunner;

class FalseColorModelPayload : public QObject
{
	Q_OBJECT
public:
	FalseColorModelPayload(FalseColoring::Type coloringType,
						   SharedMultiImgPtr img,
						   SharedMultiImgPtr grad
						   )
		: canceled(false),
		  coloringType(coloringType),
		  img(img), grad(grad),
		  runner(NULL)
	{}

	/** Start calculation.
	 *
	 * Start new thread or whatever is necessary.
	 */
	void run();

	/** Cancel computation: Actually signal the running thread. */
	void cancel();

	QPixmap getResult() { return result; }

signals:
	/** Computation progress changed. */
	void progressChanged(FalseColoring::Type coloringType, int percent);

	/** Computation completed. */
	void finished(FalseColoring::Type coloringType, bool success = true);
private slots:
	void processRunnerSuccess(std::map<std::string, boost::any> output);
	void processRunnerFailure();
	void processRunnerProgress(int percent);
private:
	bool canceled;
	FalseColoring::Type coloringType;
	SharedMultiImgPtr img;
	SharedMultiImgPtr grad;
	CommandRunner *runner;
	QPixmap result;
};


#endif // FALSECOLORMODELPAYLOAD_H

