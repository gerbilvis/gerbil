#ifndef FOLDINGBAR_H
#define FOLDINGBAR_H

#include "ui_foldingbar.h"
#include <QWidget>
#include <QMouseEvent>
#include <QVector>
#include <QPixmap>

class FoldingBar : public QWidget, private Ui::FoldingBar
{
    Q_OBJECT

public:
    explicit FoldingBar(QWidget *parent = 0);

	void fold();
	void unfold();

	void setTitle(const QString &title) { titleLabel->setText(title); }

signals:
	void toggleFold();

protected:
	void enterEvent(QEvent *);
	void leaveEvent(QEvent *);
	void mouseReleaseEvent(QMouseEvent *);

	void changeEvent(QEvent *e);

	QVector<QPixmap> arrows;
	bool folded;
	QVector<QPalette> palettes;
};

#endif // FOLDINGBAR_H
