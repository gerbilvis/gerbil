/*
	Copyright(c) 2012 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MULTI_IMG_VIEWER_H
#define MULTI_IMG_VIEWER_H

#include "ui_multi_img_viewer.h"
#include "viewport.h"
#include "viewportcontrol.h"

#include "../gerbil_gui_debug.h"

#include <multi_img.h>

#include <shared_data.h>
#include <background_task.h>
#include <background_task_queue.h>

#include <vector>
#include <map>
#include <QMenu>

class multi_img_viewer : public QWidget, private Ui::multi_img_viewer {
    Q_OBJECT
public:
	multi_img_viewer(QWidget *parent = 0);
	~multi_img_viewer();

	SharedMultiImgPtr getImage() { return image; }
	void resetImage() { image.reset(); }
	Viewport* getViewport() { return viewport; }
	void activateViewport() { viewport->activate(); }
	const multi_img::Mask& getMask() { return maskholder; }
	int getSelection() { return viewport->selection; }
	void setSelection(int band) { viewport->selection = band; }
	representation getType() { return type; }
	void setType(representation type);
	bool isPayloadHidden() { return viewportGV->isHidden(); }

	BackgroundTaskQueue *queue;
	cv::Mat1s labels;

public slots:
	void updateMask(int dim);
	void subPixels(const std::map<std::pair<int, int>, short> &points);
	void addPixels(const std::map<std::pair<int, int>, short> &points);
	void subImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void addImage(sets_ptr temp, const std::vector<cv::Rect> &regions, cv::Rect roi);
	void setImage(SharedMultiImgPtr image, cv::Rect roi);
	void setIlluminant(const std::vector<multi_img::Value> &, bool for_real);
	void changeBinCount(int bins);
	void updateBinning(int bins);
	void finishBinCountChange(bool success);
	void subLabelMask(sets_ptr temp, const cv::Mat1b &mask);
	void addLabelMask(sets_ptr temp, const cv::Mat1b &mask);
	void updateLabels();
	void toggleFold();
	void toggleLabeled(bool toggle);
	void toggleUnlabeled(bool toggle);
	void toggleLabels(bool toggle);
	void toggleLimiters(bool toggle);
	void overlay(int x, int y);
	void setActive()	{ viewport->active = true; viewport->update(); }
	void setInactive()	{  viewport->active = false; viewport->update(); }
	void updateLabelColors(QVector<QColor> labelColors, bool changed);

signals:
	void newOverlay();
	void folding();
	void setGUIEnabled(bool enable, TaskType tt);
	void toggleViewer(bool enable, representation type);
	void finishTask(bool success);

protected:
    void changeEvent(QEvent *e);

	// helpers for createMask
	void fillMaskSingle(int dim, int sel);
	void fillMaskLimiters(const std::vector<std::pair<int, int> >& limits);
	void updateMaskLimiters(const std::vector<std::pair<int, int> >&, int dim);
	void setTitle(representation type, multi_img::Value min, multi_img::Value max);

	Viewport *viewport;
	ViewportControl *control;
	SharedMultiImgPtr image;
	representation type;
	std::vector<multi_img::Value> illuminant;
	bool ignoreLabels;
	multi_img::Mask maskholder;
	bool maskValid;
	bool maskReset;
	bool titleReset;

protected slots:
	void render(bool necessary = true);

};

#endif // MULTI_IMG_VIEWER_H
