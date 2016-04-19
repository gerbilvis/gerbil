#ifndef SPECSIMTBB_H
#define SPECSIMTBB_H

#include <multi_img.h>
#include <background_task/background_task.h>
#include <shared_data.h>

#include "sm_config.h"
#include "similarity_measure.h"
#include "tbb/task_group.h"

using SimMeasure = similarity_measures::SimilarityMeasure<multi_img::Value>;

class SpecSimTbb : public BackgroundTask {

public:
    SpecSimTbb(SharedMultiImgPtr multi, qimage_ptr image,
    cv::Point coord, std::shared_ptr<SimMeasure> distfun)
        : BackgroundTask(), multi(multi), image(image),
        coord(coord), distfun(distfun) {}
    virtual ~SpecSimTbb() {}
    virtual bool run();

protected:
    tbb::task_group_context stopper;

    SharedMultiImgPtr multi;
    qimage_ptr image;
    cv::Point coord;
    std::shared_ptr<SimMeasure> distfun;
};

#endif // SPECSIMTBB_H
