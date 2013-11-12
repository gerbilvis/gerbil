#ifndef DISTVIEWGUI_H
#define DISTVIEWGUI_H

#include "ui_distviewgui.h"
#include "ui_viewportcontrol.h"

#include "viewport.h"
#include <QMenu>

class DistViewController;

class DistViewGUI : public QObject
{
	Q_OBJECT
public:
	explicit DistViewGUI(representation::t type);
	void initSignals(DistViewController *chief);

	QWidget* getFrame() { return frame; }

	vpctx_ptr getContext() { return vp->ctx; }
	sets_ptr getBinSets() { return vp->sets; }

	int getBinCount() { return uivc->binSlider->value(); }

	void setTitle(representation::t type);
	void setTitle(representation::t type,
				  multi_img::Value min, multi_img::Value max);

	// not a signal as needs to be directly called for each GUI component
	void insertPixelOverlay(const QPolygonF &points);

	static QIcon colorIcon(const QColor &color);

public slots:
	void setActive()	{ vp->activate(); vp->update(); }
	void setInactive()	{ vp->active = false; vp->update(); }
	void setEnabled(bool enabled);
	void rebuild()		{ vp->rebuild(); }

	// fold in or out
	void toggleFold();

	// from outside
	void updateLabelColors(QVector<QColor> colors);
	void toggleSingleLabel(bool toggle);

	// from our GUI
	void setAlpha(int alpha);
	void setBinLabel(int n);
	void setBinCount(int n);
	void showLimiterMenu();

signals:
	// from GUI elements to controller
	void activated();
	void folding();
	void bandSelected(int dim);
	void requestOverlay(int dim, int bin);
	void requestOverlay(const std::vector<std::pair<int, int> >& limiters,
						int dim);

	void requestBinCount(representation::t type, int bins);

	// from controller to GUI elements
	void newIlluminantCurve(QVector<multi_img::Value>);
	void toggleIlluminationShown(bool show);
	void newIlluminantApplied(QVector<multi_img::Value>);

protected:
	// initialize target, vp, ui::gv
	void initVP();
	// initialize vc, uivc
	void initVC(representation::t type);
	// initialize topbar, title
	void initTop();
	// (re-)create the menu according to labels
	void createLimiterMenu();

	representation::t type;

	QWidget *frame;
	Ui::DistViewGUI *ui;
	Ui::ViewportControl *uivc;
	Viewport *vp;
	AutohideWidget *vc;

	QVector<QColor> labelColors;
	QMenu limiterMenu;
};

#endif // DISTVIEWGUI_H
