/*	
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef MULTI_IMG_TASKS_H
#define MULTI_IMG_TASKS_H

#include "background_task.h"
#include "shared_data.h"
#include <multi_img.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace MultiImg {

class BgrSerial : public BackgroundTask {
public:
    BgrSerial(multi_img_ptr multi, mat_vec3f_ptr bgr) 
		: multi(multi), bgr(bgr) {}
	virtual ~BgrSerial() {};
    virtual void run();
    virtual void cancel() {}
protected:
    multi_img_ptr multi;
	mat_vec3f_ptr bgr;
};

}

#endif
