#ifndef MULTIIMGMODEL_H
#define MULTIIMGMODEL_H

class MultiImgModel
{
public:
    /*!
     * \brief MultiImgModel
     * \param image MultiImgModel takes ownership of the pointer.
     */
    MultiImgModel(multi_img_base *image);
    ~MultiImgModel();

    /*! \brief Locks the multi_img object and returns a guarded pointer to it. */
    GuardedMultiImgBase getImage();

private:
    multi_img_base* m_img;
    boost::recursive_mutex m_img_mutex;
};

#endif // MULTIIMGMODEL_H
