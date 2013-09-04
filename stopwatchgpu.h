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
			if (glewInit() == GLEW_OK)
				std::cerr << "there we have glew!" << std::endl;
			else
				std::cerr << "there we have no glew!" << std::endl;
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
		std::cerr << ident.toStdString() << "\t"
				  << (timerEnd - timerStart) / 1000000.0 << std::endl;
		running = false;
	}

private:
	bool running;
	QString ident;
	GLuint query[2];
};

#endif // STOPWATCHGPU_H
