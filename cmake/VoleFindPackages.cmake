# Library versions

# 2.4 introduces some new features we use and older versions are buggy, too!
set(VOLE_MINIMUM_OPENCV_VERSION "2.4.0")

# 4.7 introduces some GL features we use
set(VOLE_MINIMUM_QT_VERSION "4.7.0")

# 1.47 introduces CHRONO, and I am tired of even guarding lib dependencies!
set(VOLE_MINIMUM_BOOST_VERSION "1.47")

# OpenCV
find_package(OpenCV PATHS "/net/cv/lib/share/OpenCV" "/local/opencv/share/OpenCV")
if(OpenCV_FOUND)
	if(NOT (${OpenCV_VERSION} VERSION_LESS ${VOLE_MINIMUM_OPENCV_VERSION}))
		message(STATUS "Found OpenCV version: "
			"${OpenCV_VERSION} (minimum required: ${VOLE_MINIMUM_OPENCV_VERSION})")
	else()
		message(SEND_ERROR "Unsupported OpenCV version: "
			"${OpenCV_VERSION} (minimum required: ${VOLE_MINIMUM_OPENCV_VERSION})")
		# cmake configure will fail after SEND_ERROR,
		# no need to set OpenCV_FOUND FALSE.
	endif()
	add_definitions(-DOPENCV_VERSION=${OpenCV_VERSION})
	add_definitions(-DWITH_OPENCV2)
else()
	# OpenCV_FOUND==FALSE is handled by vole_check_package
endif()
vole_check_package(OPENCV
	"OpenCV"
	"Please install OpenCV >=${VOLE_MINIMUM_OPENCV_VERSION} or set OpenCV_DIR."
	OpenCV_FOUND
	"${OpenCV_INCLUDE_DIRS}"
	"${OpenCV_LIBS}"
)

# Thread Building Blocks
find_package(TBB)
vole_check_package(TBB
	"TBB"
	"Please install TBB"
	TBB_FOUND
	"${TBB_INCLUDE_DIRS}"
	"${TBB_LIBRARIES}"
)

# OpenGL
find_package(OpenGL)
vole_check_package(OPENGL
	"OpenGL"
	"Please check your OpenGL installation."
	OpenGL_FOUND
	"${OpenGL_INCLUDE_DIR}"
	"${OpenGL_LIBRARIES}"
)

# GLEW
#find_package(GLEW)
#vole_check_package(GLEW
#	"GLEW"
#	"Please install GLEW."
#	GLEW_FOUND
#	"${GLEW_INCLUDE_PATH}"
#	"${GLEW_LIBRARIES}"
#)

## We are not quite there yet to support Qt5 in the code (run-time problems)
## QtWidgets 5
#find_package(Qt5Widgets)
#vole_check_package(QT
#	"Qt5"
#	"Please install Qt5 OR Qt4 >=${VOLE_MINIMUM_QT_VERSION} or set QT_QMAKE_EXECUTABLE."
#	Qt5Widgets_FOUND
#	"${Qt5Widgets_INCLUDE_DIRS}"
#	"${Qt5Widgets_LIBRARIES}"
#)

## QtOpenGL 5 (which is a backwards-compatibility module)
#find_package(Qt5OpenGL)
#vole_check_package(QT_OPENGL
#	"Qt5 OpenGL"
#	"Please install Qt5 OR Qt4 >=${VOLE_MINIMUM_QT_VERSION} or set QT_QMAKE_EXECUTABLE."
#	Qt5OpenGL_FOUND
#	"${Qt5OpenGL_INCLUDE_DIRS}"
#	"${Qt5OpenGL_LIBRARIES}"
#)
#
#if(Qt5Widgets_FOUND)
#	set(WITH_QT5 TRUE)
#else()
	# QtGui 4
	find_package(Qt4 ${VOLE_MINIMUM_QT_VERSION} COMPONENTS QtCore QtGui)
	vole_check_package(QT
		"Qt4"
		"Please install Qt4 >=${VOLE_MINIMUM_QT_VERSION} or set QT_QMAKE_EXECUTABLE."
		QT_FOUND
		"${QT_INCLUDE_DIR};${QT_QTCORE_INCLUDE_DIR};${QT_QTGUI_INCLUDE_DIR}"
		"${QT_QTCORE_LIBRARY};${QT_QTGUI_LIBRARY}"
	)

	# QtOpenGL 4
	find_package(Qt4 ${VOLE_MINIMUM_QT_VERSION} COMPONENTS QtOpenGL)
	vole_check_package(QT_OPENGL
		"Qt4 OpenGL"
	"Please install Qt4 >=${VOLE_MINIMUM_QT_VERSION} or set QT_QMAKE_EXECUTABLE."
		QT_QTOPENGL_FOUND
		"${QT_INCLUDE_DIR};${QT_QTOPENGL_INCLUDE_DIR}"
		"${QT_QTOPENGL_LIBRARY}"
	)

	# QtXml 4
	find_package(Qt4 ${VOLE_MINIMUM_QT_VERSION} COMPONENTS QtXml)
	vole_check_package(QT_XML
		"Qt4 XML"
	"Please install Qt4 >=${VOLE_MINIMUM_QT_VERSION} or set QT_QMAKE_EXECUTABLE."
		QT_QTXML_FOUND
		"${QT_INCLUDE_DIR};${QT_QTXML_INCLUDE_DIR}"
		"${QT_QTXML_LIBRARY}"
	)

# Boost
if(WIN32)
	set(Boost_USE_STATIC_LIBS ON)
	set(BOOST_ROOT "C:\\boost" CACHE PATH "Boost Root Directory.")
else()
	set(BOOST_ROOT "" CACHE PATH "Boost Root Directory.")
endif()
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION})
# if you need the boost version in the code, #include <boost/version.hpp>
#if(Boost_FOUND)
#	add_definitions(-DBOOST_VERSION=${Boost_VERSION})
#endif()
vole_check_package(BOOST
	"Boost"
	"Please install Boost system >= ${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	""
)

# Boost system
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS system)
vole_check_package(BOOST_SYSTEM
	"Boost system"
	"Please install Boost system >= ${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_SYSTEM_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_SYSTEM_LIBRARY}"
)

# Boost filesystem
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS filesystem)
vole_check_package(BOOST_FILESYSTEM
	"Boost filesystem"
	"Please install Boost filesystem >= ${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_FILESYSTEM_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_FILESYSTEM_LIBRARY}"
)

# Boost program options
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS program_options)
vole_check_package(BOOST_PROGRAM_OPTIONS
	"Boost program options"
	"Please install Boost program options >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_PROGRAM_OPTIONS_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_PROGRAM_OPTIONS_LIBRARY}"
)

# Boost serialization
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS serialization)
vole_check_package(BOOST_SERIALIZATION
	"Boost Serialization"
	"Please install Boost serialization >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_SERIALIZATION_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_SERIALIZATION_LIBRARY}"
)

# Boost thread
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS thread)
vole_check_package(BOOST_THREAD
	"Boost thread"
	"Please install Boost thread >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_THREAD_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_THREAD_LIBRARY}"
)

# Boost date time
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS date_time)
vole_check_package(BOOST_DATE_TIME
	"Boost date time"
	"Please install Boost date time >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_DATE_TIME_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_DATE_TIME_LIBRARY}"
)

# Boost chrono (needed by boost_thread, we explicitely depend for Windows builds)
find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS chrono)
vole_check_package(BOOST_CHRONO
	"Boost chrono"
	"Please install Boost chrono >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
	Boost_CHRONO_FOUND
	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
	"${Boost_CHRONO_LIBRARY}"
)

# Boost python
#find_package(Boost ${VOLE_MINIMUM_BOOST_VERSION} COMPONENTS python)
#vole_check_package(BOOST_PYTHON
#	"Boost python"
#	"Please install Boost python >=${VOLE_MINIMUM_BOOST_VERSION} or set Boost_ROOT."
#	Boost_PYTHON_FOUND
#	"${Boost_INCLUDE_DIR}/include/;${Boost_INCLUDE_DIR}"
#	"${Boost_PYTHON_LIBRARY}"
#)

# PETSc && SLEPc
#find_package(PETSc)
#vole_check_package(PETSC
#	"PETSc"
#	"Please install PETSc (and SLEPc) or set PETSC_DIR."
#	PETSC_FOUND
#	"${PETSC_INCLUDE_DIR}"
#	"${PETSC_LIBRARIES}"
#)
#find_package(SLEPc)
#vole_check_package(SLEPC
#	"SLEPc"
#	"Please install SLEPc >=${VOLE_MINIMUM_SLEPC_VERSION} or set SLEPC_DIR."
#	SLEPC_FOUND
#	"${SLEPC_INCLUDE_DIR}"
#	"${SLEPC_LIBRARIES}"
#)

# libEigen3
#find_package(Eigen3 ${VOLE_MINIMUM_EIGEN_VERSION})
#vole_check_package(EIGEN3
#	"Eigen"
#	"Please install libeigen3 >=${VOLE_MINIMUM_EIGEN_VERSION}"
#	EIGEN3_FOUND
#	"${EIGEN3_INCLUDE_DIR}"
#	""
#)

# GDAL
find_package(GDAL)
vole_check_package(GDAL
	"GDAL"
	"Please install GDAL"
	GDAL_FOUND
        "${GDAL_INCLUDE_DIR}"
        "${GDAL_LIBRARIES}"
)
