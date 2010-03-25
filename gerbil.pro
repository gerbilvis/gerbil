TEMPLATE = app

CONFIG += warn_off \
    debug_and_release
CONFIG(debug, debug|release):TARGET = gerbil_dbg
else:TARGET = gerbil

# OPENCV
CONFIG += link_pkgconfig
PKGCONFIG += opencv

# BOOST
LIBS += -lboost_program_options

# OPENGL
QT += opengl
QMAKE_CXXFLAGS += -Wall # -fopenmp -DGLIBCXX_PARALLEL
QMAKE_CXXFLAGS_RELEASE = -march=i686 \
    -O3

# Input
# FORMS =
HEADERS += auxiliary.h \
    mfams.h \
    multi_img.h \
    multi_img_viewer.h \
    viewport.h
SOURCES += main.cpp \
    multi_img.cpp \
    mfams.cpp \
    auxiliary.cpp \
    io.cpp \
    multi_img_viewer.cpp \
    viewport.cpp
FORMS += multi_img_viewer.ui
