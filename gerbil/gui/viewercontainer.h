#ifndef VIEWERCONTAINER_H
#define VIEWERCONTAINER_H

#include <QWidget>

#include "multi_img_viewer.h"

//namespace Ui {
//class ViewersWidget;
//}

class ViewerContainer : public QWidget
{
    Q_OBJECT
protected:
    typedef QList<multi_img_viewer*> ViewerList;
    typedef QMultiMap<representation, multi_img_viewer*> ViewerMultiMap;
    
public:
    explicit ViewerContainer(QWidget *parent = 0);
    ~ViewerContainer();

private:


protected:
    ViewerList vl;
    ViewerMultiMap vm;

    QLayout *vLayout;

    void initUi();
    /*! \brief Add a multi_img_viewer to the widget.
     *
     * The viewer will be inserted at the bottom of the vertical layout.
     */
    multi_img_viewer *createViewer(representation rep);
//private:
//    Ui::ViewersWidget *ui;
};

#endif // VIEWERCONTAINER_H
