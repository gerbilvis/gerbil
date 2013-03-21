
#include <boost/thread/recursive_mutex.hpp>
#include <stdexcept>
#include <multi_img.h>

#include "guarded_data.h"
#include "multiimgmodel.h"

MultiImgModel::MultiImgModel(multi_img_base *image)
    : m_img(image)
{
}

MultiImgModel::~MultiImgModel()
{
    delete m_img;
}

GuardedMultiImgBase MultiImgModel::getImage()
{
    return GuardedMultiImgBase(*m_img, m_img_mutex);
}
