#ifndef GERBIL_APP_SUPPORT_H
#define GERBIL_APP_SUPPORT_H

#include <multi_img.h>

#include <string>
#include <utility>
#include <vector>

/** \file Helper functions for GerbilApplication. */


/**  Initialize OpenCV state.
 *
 * All OpenCV functions that are called from parallelized parts of gerbil have
 * to be first executed in single-threaded environment. This is actually
 * required only for functions that contain 'static const' variables, but to
 * avoid investigation of OpenCV sources and to defend against any future
 * changes in OpenCV, it is advised not to omit any used function. Note that
 * 'static const' variables within functions are initialized in a lazy manner
 * and such initialization is not thread-safe because setting the value and
 * init flag of the variable is not an atomic operation.
 */
void init_opencv();

/** Initialize CUDA. */
void init_cuda();

/** Check for system requirements.
 *
 * Checks for MXX, SSE and SSE2 and for required OpenGL features.
 *
 * @return Returns true if all requirements are met, otherwise false.
 */
bool check_system_requirements();

/** Register types with Qt type system. */
 void registerQMetaTypes();

/** Tests wether or not to load the multi_img in limited mode.
 *
 * Opens dialog for querying the user. Calls exit() if user decides to close
 * the application.
 *
 * @return true if multi_img should be loaded in limited mode, otherwise false.
 */
// FIXME use QString for filenames (full unicode support).
bool determine_limited(const std::pair<std::vector<std::string>,
							  std::vector<multi_img::BandDesc> > &filelist);

#endif // GERBIL_APP_SUPPORT_H
