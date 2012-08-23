find_path(
	QHULL_INCLUDE_DIR
	NAMES qhull/qhull_a.h
	HINTS
	PATH_SUFFIXES include
	PATHS
	/usr/local
	/usr
	/opt/local
)

find_library(
	QHULL_LIBRARIES
	NAMES qhull
	HINTS
	PATH_SUFFIXES lib64 lib
	PATHS
	/usr/local
	/usr
	/opt/local
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QHULL DEFAULT_MSG QHULL_LIBRARIES QHULL_INCLUDE_DIR)
mark_as_advanced(QHULL_INCLUDE_DIR QHULL_LIBRARIES)
