#ifndef BAND2QIMAGETBB_H
#define BAND2QIMAGETBB_H

#include <multi_img.h>
#include <background_task/background_task.h>
#include <shared_data.h>

#include <tbb/blocked_range2d.h>
#include <tbb/task_group.h>

class Band2QImage {

public:
    Band2QImage(multi_img::Band &band, QImage &image,
        multi_img::Value minval, multi_img::Value maxval)
        : band(band), image(image), minval(minval), maxval(maxval) {}
    void operator()(const tbb::blocked_range2d<int> &r) const;

private:
    multi_img::Band &band;
    QImage &image;
    multi_img::Value minval;
    multi_img::Value maxval;
};

class Band2QImageTbb : public BackgroundTask {

public:
    Band2QImageTbb(SharedMultiImgPtr multi, qimage_ptr image, size_t band)
        : BackgroundTask(), multi(multi), image(image), band(band) {}
    virtual ~Band2QImageTbb() {}
    virtual bool run();
    virtual void cancel() { stopper.cancel_group_execution(); }

protected:
    tbb::task_group_context stopper;

    SharedMultiImgPtr multi;
    qimage_ptr image;
    size_t band;
};

#endif // BAND2QIMAGETBB_H
