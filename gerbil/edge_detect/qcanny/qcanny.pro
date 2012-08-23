TEMPLATE = app

CONFIG -= release

CONFIG += debug

OBJECTS_DIR = /home/knorx/UNI/Studienarbeit/working_directory/code/qcanny/qcanny/obj

MOC_DIR = /home/knorx/UNI/Studienarbeit/working_directory/code/qcanny/qcanny/moc

SOURCES += main.cpp \
qcanny.cpp \
myCanny.cpp
HEADERS += qcanny.h \
_cvgeom.h \
_cvimgproc.h \
myCanny.h
INCLUDEPATH += /usr/local/include/opencv/

LIBS += -L/usr/local/lib \
  -L/usr/local/lib/ \
  -lopencv_highgui \
  -lopencv_core

RESOURCES += qcanny.qrc

