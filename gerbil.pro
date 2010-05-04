TEMPLATE = app

# only for the editor.. should not be here!
INCLUDEPATH += /net/cv/lib32/include/opencv
CONFIG += warn_off \
    debug_and_release
CONFIG(debug, debug|release):TARGET = gerbil_dbg
else:TARGET = gerbil

# OPENCV
CONFIG += link_pkgconfig
PKGCONFIG += opencv

# BOOST
LIBS += -lboost_program_options -lboost_filesystem

# OPENGL
QT += opengl
QMAKE_CXXFLAGS += -Wall -DVOLE_WITH_BOOST # -fopenmp -DGLIBCXX_PARALLEL
QMAKE_CXXFLAGS_RELEASE = -march=i686 \
    -O3

# Input
HEADERS += auxiliary.h \
    mfams.h \
    multi_img.h \
    multi_img_viewer.h \
    viewport.h \
    viewerwindow.h
SOURCES += main.cpp \
    multi_img.cpp \
    mfams.cpp \
    auxiliary.cpp \
    io.cpp \
    multi_img_viewer.cpp \
    viewport.cpp \
    viewerwindow.cpp
FORMS += multi_img_viewer.ui \
    viewerwindow.ui
RESOURCES += gerbil.qrc
