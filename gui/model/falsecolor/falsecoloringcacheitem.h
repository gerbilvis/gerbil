#ifndef FALSECOLORINGCACHEITEM_H
#define FALSECOLORINGCACHEITEM_H

/** Cache item for computed false color images. */
class FalseColoringCacheItem {
public:
	FalseColoringCacheItem() : pixmap_(), mat_(nullptr) {}  // invalid cache item
	FalseColoringCacheItem(QPixmap img, cv::Mat3f mat) : pixmap_(img), mat_(new cv::Mat3f(mat))  {} // valid cache item

	// Default copy constructor and assignment operator.

	void invalidate() { pixmap_ = QPixmap(); }
	bool valid() { return ! pixmap_.isNull(); }
	QPixmap pixmap() { return pixmap_; }
	cv::Mat3f* mat() { return mat_; }

private:
	QPixmap pixmap_;
	cv::Mat3f* mat_;
};

#endif // FALSECOLORINGCACHEITEM_H
