#ifndef GERBIL_OSTREAM_OPS_H
#define GERBIL_OSTREAM_OPS_H

#ifdef WITH_QT
#ifdef WITH_OPENCV

#include <iosfwd>
#include <opencv2/core/core.hpp>

class QSize;
class QPoint;
class QPointF;

std::ostream &operator<<(std::ostream& os, const QSize& size);
std::ostream& operator<<(std::ostream& os, const QPointF &p);
std::ostream &operator<<(std::ostream& os, const cv::Rect& r);

#endif
#endif
#endif // GERBIL_OSTREAM_OPS_H
