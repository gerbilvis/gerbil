#include "foldingbar.h"
#include <QPainter>
#include <iostream>

FoldingBar::FoldingBar(QWidget *parent) :
	QWidget(parent), folded(false)
{
    setupUi(this);
	// 0: folded (disabled)
	arrows.push_back(QPixmap(":/basic/arrow-right"));
	// 1: unfolded (enabled)
	arrows.push_back(QPixmap(":/basic/arrow-down"));
	// 2: mouse hover
	arrows.push_back(QPixmap(":/basic/arrow-downright"));

	/* pair of inactive/active color palettes
	 * used for recoloring on hover
	 */
	palettes.push_back(palette());
	palettes.push_back(palette());
	palettes[1].setColor(QPalette::Window,
						 palette().color(QPalette::Highlight));
	palettes[1].setColor(QPalette::WindowText,
						 palette().color(QPalette::HighlightedText));
	palettes[1].setColor(QPalette::Text,
						 palette().color(QPalette::HighlightedText));

	// child widgets should pass-through click events to us
	arrowLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
	titleLabel->setAttribute(Qt::WA_TransparentForMouseEvents);
}

void FoldingBar::fold()
{
	folded = true;
	titleLabel->setEnabled(false);
	arrowLabel->setPixmap(arrows[0]);
}

void FoldingBar::unfold()
{
	folded = false;
	titleLabel->setEnabled(true);
	arrowLabel->setPixmap(arrows[1]);
}

void FoldingBar::enterEvent(QEvent *)
{
	arrowLabel->setPixmap(arrows[2]);
	titleLabel->setPalette(palettes[1]);
	setPalette(palettes[1]);
}

void FoldingBar::leaveEvent(QEvent *){
	arrowLabel->setPixmap(arrows[(folded ? 0 : 1)]);
	titleLabel->setPalette(palettes[0]);
	setPalette(palettes[0]);
}

void FoldingBar::mouseReleaseEvent(QMouseEvent *ev)
{
	emit toggleFold();
}

void FoldingBar::changeEvent(QEvent *e)
{
    QWidget::changeEvent(e);
	switch (e->type()) {
    case QEvent::LanguageChange:
		/* strange behavior: when a dialog is opened, this is performed, and
		 * it will reset the text we customly set in the titleLabel! */
		// retranslateUi(this);
        break;
    default:
        break;
	}
}
