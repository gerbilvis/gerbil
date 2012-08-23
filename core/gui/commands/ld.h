#ifndef LD_H
#define LD_H

#include "lighting_direction_core.h"

#include <QWidget>
#include <QScrollBar>
#include <QLabel>
#include <QTabWidget>
#include <QAction>
#include <QPushButton>
#include <QMessageBox>
#include <QCloseEvent>
#include <QFileDialog>
#include <QResizeEvent>
#include <QSignalMapper>
#include <QSlider>
#include <QPainter>

#include "command_wrapper.h"
#include "image_plane.h"
#include "executionthread.h"
#include "blending_image.h"


class ld : public CommandWrapper
{
	Q_OBJECT

public slots:
	void    startProcess();
//    void  imgManipulationStart( void );
//    void  imgManipulationDone( QImage );
//	void    vsldAlphaChanged(int);
//	void	closeEvent( QCloseEvent* );
	void  alphaChanged(int);

public:
	ld(VoleGui *parent, std::pair<QString, QImage*> img);
	~ld();
	void resizeGUI(QResizeEvent*);
	BlendingImage *blendingImage;

	static CommandWrapper* spawn(VoleGui *parent);
	
private:
// popup menu
	QAction       *rect0Act, *rect1Act, *rect2Act, *rect3Act, *remAct;
	QSignalMapper *sigMap;
	QMenu *popMenu;

// original image info
	QImage *kantenImg;

// lighting direction computation class;
	LightingDirectionCore *ldc;

// gui elements
	QWidget     *selTab, *resTab;
	QPushButton *btnExecute, *btnZoomIn, *btnZoomOut;
	QSlider     *vsldAlpha;
	ImagePlane  *imagePlane;

	void        createGUI(QWidget*);
	QWidget*    createSelectionTab(void);
	void        createActions(void);
	void        createPopupMenu(void);

};

#endif // LD_H
