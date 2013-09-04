#ifndef STOPWATCHGPU_H
#define STOPWATCHGPU_H

#include <GL/glew.h>
#include <QString>
#include <iostream>

class StopwatchGPU {

public:
	StopwatchGPU(QString ident) : running(false), ident(ident)
	{
		// init GLEW on very first call. Do it here because we need the GL context.
		initGLEW();
		glGenQueries(2, query);
		start();
	}
	~StopwatchGPU()
	{
		stop();
		glDeleteQueries(2, query);
	}
	static void initGLEW()
	{
		static bool glewInitialized = false;
		if (!glewInitialized) {
			if (glewInit() != GLEW_OK)
				std::cerr << "GLEW initialization failed! Run!" << std::endl;
			glewInitialized = true;
		}
	}

	void start()
	{
		glQueryCounter(query[0], GL_TIMESTAMP);
		running = true;
	}
	void stop()
	{
		if (!running)
			return;

		glQueryCounter(query[1], GL_TIMESTAMP);
		GLint done = 0;
		while (!done) {
			glGetQueryObjectiv(query[1], GL_QUERY_RESULT_AVAILABLE, &done);
		}
		GLuint64 timerStart, timerEnd;
		glGetQueryObjectui64v(query[0], GL_QUERY_RESULT, &timerStart);
		glGetQueryObjectui64v(query[1], GL_QUERY_RESULT, &timerEnd);
		double measure = (timerEnd - timerStart) / 1000000000.0;

		// invalid measure (timing way too high, event was just too short)
		if (measure > 600) {
			// just print nothing
			running = false;
			return;
		}

		std::ios_base::fmtflags orig_flags = std::cerr.flags();
		std::streamsize orig_precision = std::cerr.precision();
		std::cerr.setf(std::ios_base::fixed);
		std::cerr.precision(6);
		std::cerr.width(10);
		std::cerr << measure << " s: " << ident.toStdString() << std::endl;
		std::cerr.flags(orig_flags); std::cerr.precision(orig_precision);

		running = false;
	}

private:
	bool running;
	QString ident;
	GLuint query[2];
};

#endif // STOPWATCHGPU_H
