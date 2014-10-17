#include "recentfiledelegate.h"
#include "recentfile.h"

#include <iostream>

#include <QPainter>
#include <QStaticText>
#include <QFontMetrics>

#define GGDBG_MODULE
#include <gerbil_gui_debug.h>

static const QFont::Weight fileNameFontWeight = QFont::Bold;
static const int margin = 2; // in item rect
static const int iconSpacing = 8; // horizontally between icon and text
static const QSize spacedIconSize(
		RecentFile::iconSize() + QSize(iconSpacing, 0));


RecentFileDelegate::RecentFileDelegate(QObject *parent)
	: QStyledItemDelegate(parent)
{
}

RecentFileDelegate::~RecentFileDelegate()
{
}

void RecentFileDelegate::paint(QPainter *painter,
							   const QStyleOptionViewItem &option,
							   const QModelIndex &index) const
{
	if (index.data(RecentFile::RecentFileDataRole).isNull()) {
		QStyledItemDelegate::paint(painter, option, index);
		return;
	}

	RecentFile rf = recentFile(index);

	painter->save();
	// draw background (yes, it's that hard)
	if (option.state & QStyle::State_Selected &&
			! bool(option.state & QStyle::State_Active) )
	{
		// inactive
		painter->fillRect(option.rect,
						  option.palette.brush(
							  QPalette::Inactive,
							  QPalette::Highlight));
	} else if ( option.state & QStyle::State_Selected ) {
		// active
		painter->fillRect(option.rect,
						  option.palette.brush(
							  QPalette::Active,
							  QPalette::Highlight));
	}

	// draw icon
	QPoint p = option.rect.topLeft() + QPoint(margin, margin);
	QRect iconRect(p, RecentFile::iconSize());
	QPixmap pixmap = index.data(Qt::DecorationRole).value<QPixmap>();
	if (! pixmap.isNull()) {
		// center in icon frame
		QSize off =
				( pixmap.size().boundedTo(RecentFile::iconSize()) -
				  RecentFile::iconSize() ) / 2;
		p -= QPoint(off.width(), off.height());
		iconRect = QRect(p, pixmap.size().boundedTo(RecentFile::iconSize()));
		painter->drawPixmap(p, pixmap);
	} else {
		painter->drawLine(iconRect.topLeft(), iconRect.bottomRight());
		painter->drawLine(iconRect.bottomLeft(), iconRect.topRight());
	}
	painter->drawRect(iconRect);

	// draw text
	painter->save();
	QFont font(option.font);
	QFontMetrics fontMetrics(font);
	QFont fileNameFont(option.font);
	QFontMetrics fileNameFontMetrics(fileNameFont);
	fileNameFont.setWeight(QFont::Bold);
	painter->setFont(fileNameFont);
	int txoff = margin+spacedIconSize.width();
	// center vertically
	int tyoff = (option.rect.height() -
				 fileNameFontMetrics.lineSpacing() -
				 2 * fontMetrics.lineSpacing()) / 2;
	painter->drawText(option.rect.adjusted(txoff, tyoff, txoff, tyoff),
					  "File: " + rf.getFileNameWithoutPath());
	tyoff += fileNameFontMetrics.lineSpacing();
	painter->restore();
	painter->setFont(option.font);
	painter->drawText(option.rect.adjusted(txoff, tyoff, txoff, tyoff),
					  "Path: " + rf.fileName + "\n" +
					  "Last opened: " + rf.lastOpenedString());
	painter->restore();
}

QSize RecentFileDelegate::sizeHint(const QStyleOptionViewItem &option,
								   const QModelIndex &index) const
{
	if (index.data(RecentFile::RecentFileDataRole).isNull()) {
		return QStyledItemDelegate::sizeHint(option, index);
	}

	RecentFile rf = recentFile(index);
	QFont fileNameFont(option.font);
	fileNameFont.setWeight(QFont::Bold);
	QFontMetrics fm(fileNameFont);
	QSize textSize;
	textSize = textSize.expandedTo(
				fm.boundingRect("File: " + rf.getFileNameWithoutPath()).size());
	int fileNameTextHeight = textSize.height();
	fm = QFontMetrics(option.font);
	textSize = textSize.expandedTo(
				fm.boundingRect("Path: " + rf.fileName).size());;
	textSize = textSize.expandedTo(
				fm.boundingRect("Last opened: " + rf.lastOpenedString()).size());
	int textHeight = textSize.height();
	textSize.rheight() = fileNameTextHeight + 2 * textHeight;

	QSize size;
	size.rheight() = qMax(spacedIconSize.height(), textSize.height())
			+ 2 * margin + 1;
	size.rwidth() = spacedIconSize.width() +  textSize.width() + 2 * margin;
	return size;
}

