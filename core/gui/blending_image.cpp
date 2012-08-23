#include <QDebug>
#include <QPainter>
#include <QMouseEvent>
#include <QMessageBox>

#include "blending_image.h"

#include <iostream>

BlendingImage::BlendingImage()
{
	isActive = true;
	locked = true;
}

BlendingImage::BlendingImage(QImage blendedImage, qreal alpha)
{
	isActive = true;
	init(blendedImage, alpha);
}

void BlendingImage::deinit() {
	locked = true;
}

void BlendingImage::init(QImage blendedImage, qreal alpha)
{
	this->blendedImage = blendedImage;
	this->alpha = alpha;
	locked = false;
}

void BlendingImage::alphaChanged(int value) {
	qreal alpha = 1 - (qreal)value / 100;
	this->alpha = alpha;
}

void BlendingImage::setActive(bool isActive) {
	this->isActive = isActive;
}

void BlendingImage::draw(QImage *img, QPainter *p) {
	if (!isActive || locked || (blendedImage.width() == 0)) return; // no blending if we have no image :(
	QImage scaledImage = blendedImage.scaledToHeight(img->height());
	p->setCompositionMode(QPainter::CompositionMode_SourceOver);
	p->setOpacity(alpha);
	p->drawImage(QPoint(0, 0), blendedImage);
	p->setOpacity(1);
}

void BlendingImage::setBlendedImage(QImage nImg) { blendedImage = nImg; }

BlendingImage::~BlendingImage() {}

