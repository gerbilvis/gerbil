#!/bin/bash
if [ $# -eq 0 ]; then
cat << EOF
Usage: configure.sh <OpenCV prefix> [relative path] [norpath]
e.g. # ./configure.sh /usr

You can optionally specify a relative part pointing to the sources for
building outside the source tree.
If you use the 'norpath' option, files will be built without hard-coded rpath (useful for distro packaging).

Build dependencies:
 * cmake >= 2.6
 * boost >= 1.35 (thread, program_options, filesystem)
 * Threading Building Blocks (tbb)
 * Qt >= 4.8
 * OpenCV >= 2.4
 * libGDAL (optional)

Use 'ccmake .' for further refinement of the configuration. See 'man cmake'.
Then use 'make' to build binaries into 'bin/'.
EOF
exit 1
fi

if [ "$3" = "norpath" ]; then
	skiptxt='-DCMAKE_SKIP_RPATH=on'
else
	skiptxt=''
fi

cmake -DOpenCV_DIR="$1/share/OpenCV" \
-DCMAKE_BUILD_TYPE=Release \
$skiptxt ./$2
cmake .
if [ $? -ne 0 ]; then
echo " *** Please use CMake to resolve configuration issues (e.g. 'ccmake .')."
fi
