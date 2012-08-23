/********************************************************************************
** Form generated from reading ui file 'vole_gui.ui'
**
** Created: Thu Nov 18 14:18:58 2010
**      by: Qt User Interface Compiler version 4.5.2
**
** WARNING! All changes made in this file will be lost when recompiling ui file!
********************************************************************************/

#ifndef UI_VOLE_GUI_H
#define UI_VOLE_GUI_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QStatusBar>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_VoleGui
{
public:
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QPushButton *startButton;
    QSpacerItem *horizontalSpacer;
    QTabWidget *workspaceContainer;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *VoleGui)
    {
        if (VoleGui->objectName().isEmpty())
            VoleGui->setObjectName(QString::fromUtf8("VoleGui"));
        VoleGui->resize(1024, 768);
        QSizePolicy sizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(VoleGui->sizePolicy().hasHeightForWidth());
        VoleGui->setSizePolicy(sizePolicy);
        centralWidget = new QWidget(VoleGui);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setMargin(11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        startButton = new QPushButton(centralWidget);
        startButton->setObjectName(QString::fromUtf8("startButton"));

        horizontalLayout->addWidget(startButton);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        verticalLayout->addLayout(horizontalLayout);

        workspaceContainer = new QTabWidget(centralWidget);
        workspaceContainer->setObjectName(QString::fromUtf8("workspaceContainer"));

        verticalLayout->addWidget(workspaceContainer);

        VoleGui->setCentralWidget(centralWidget);
        statusbar = new QStatusBar(VoleGui);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        VoleGui->setStatusBar(statusbar);

        retranslateUi(VoleGui);

        workspaceContainer->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(VoleGui);
    } // setupUi

    void retranslateUi(QMainWindow *VoleGui)
    {
        VoleGui->setWindowTitle(QApplication::translate("VoleGui", "Image Forensics Toolbox", 0, QApplication::UnicodeUTF8));
        startButton->setText(QApplication::translate("VoleGui", "Start new investigation...", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(VoleGui);
    } // retranslateUi

};

namespace Ui {
    class VoleGui: public Ui_VoleGui {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VOLE_GUI_H
