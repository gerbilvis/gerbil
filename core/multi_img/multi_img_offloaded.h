#ifndef MULTI_IMG_OFFLOADED_H
#define MULTI_IMG_OFFLOADED_H

#include <multi_img.h>

class multi_img_offloaded : public multi_img_base {
public:
	/// creates the multi_img with limited functionality and with bands offloaded to persistent storage
	multi_img_offloaded(const std::vector<std::string> &files, const std::vector<BandDesc> &descs);

	/// virtual destructor, does nothing
	virtual ~multi_img_offloaded() {}

	/// returns number of bands
	virtual unsigned int size() const;

	/// returns true if image is uninitialized
	virtual bool empty() const;

	/// returns one band
    virtual void getBand(size_t band, Band &data) const;

	/// returns the roi part of the given band
	virtual void scopeBand(const Band &source, const cv::Rect &roi, Band &target) const;

protected:
	std::vector<std::pair<std::string, int> > bands;

	MULTI_IMG_FRIENDS
};

#endif // MULTI_IMG_OFFLOADED_H
