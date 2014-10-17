#ifndef RECENTFILEDELEGATE_H
#define RECENTFILEDELEGATE_H

#include <QStyledItemDelegate>

class QStaticText;

/** QStyledItemDelegate class for rendering recent file entries in QListView. */
class RecentFileDelegate : public QStyledItemDelegate
{
public:
	RecentFileDelegate(QObject * parent = 0);
	~RecentFileDelegate();

	void paint(QPainter * painter,
			   const QStyleOptionViewItem & option,
			   const QModelIndex & index) const;

	QSize sizeHint (const QStyleOptionViewItem & option,
					const QModelIndex & index ) const;
};

#endif // RECENTFILEDELEGATE_H
