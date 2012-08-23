#ifndef MOUSE_OVER_IMAGE_GENERATOR_H
#define MOUSE_OVER_IMAGE_GENERATOR_H

#include <QWidget>

class MouseOverImageGenerator {
public:
	MouseOverImageGenerator() : outputXDim(-1), outputYDim(-1) { }

	MouseOverImageGenerator(int xDim, int yDim) : outputXDim(xDim), outputYDim(yDim) { }
	
	void setOutputSize(int xDim, int yDim);
	virtual QWidget *getMouseOverWidget() = 0;
	virtual void updateMouseOverWidget(int x, int y) = 0;

	virtual ~MouseOverImageGenerator() {}

protected:
	int outputXDim, outputYDim;
};

#endif // MOUSE_OVER_IMAGE_GENERATOR_H
