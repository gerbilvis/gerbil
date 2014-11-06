#include "gerbil_ostream_ops.h"

#ifdef WITH_QT
#ifdef WITH_OPENCV

#include <ostream>
#include <QSize>
#include <QPoint>
#include <boost/format.hpp>

std::ostream &operator<<(std::ostream& os, const QSize& s) {
	return os << s.width() << "x" << s.height();
}

std::ostream &operator<<(std::ostream &os, const QPointF &p) {
	os << "(" << p.x() << "," << p.y() << ")";
	return os;
}

std::ostream &operator<<(std::ostream& os, const cv::Rect& r)
{
	os << boost::format("(%1%,%2%)+%3%+%4%") % r.x % r.y % r.width % r.height;
	return os;
}

#endif
#endif
