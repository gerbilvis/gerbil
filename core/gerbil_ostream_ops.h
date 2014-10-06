#ifndef GERBIL_OSTREAM_OPS_H
#define GERBIL_OSTREAM_OPS_H

#include <iosfwd>
#include <opencv2/core/core.hpp>

class QSize;
class QPoint;
class QPointF;

std::ostream &operator<<(std::ostream& os, const QSize& size);
std::ostream& operator<<(std::ostream& os, const QPointF &p);
std::ostream &operator<<(std::ostream& os, const cv::Rect& r);

#endif // GERBIL_OSTREAM_OPS_H
