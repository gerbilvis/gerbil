#ifndef COMMAND_SEG_FELZENSZWALB_H
#define COMMAND_SEG_FELZENSZWALB_H

#include "seg_felzenszwalb_config.h"

#include <command.h>
#include <iostream>
#include <cv.h>


namespace vole {
 /** \addtogroup VoleModules */

/** Command line interface to the Felzenszwalb/Huttenlocher RGB-image segmentation
 * \ingroup VoleModules
 */
class CommandSegFelzenszwalb : public Command {
public:
	CommandSegFelzenszwalb();
	~CommandSegFelzenszwalb();
	int execute();

	void printShortHelp() const;
	void printHelp() const;

	SegFelzenszwalbConfig config;

private:
};

}

#endif // COMMAND_SEG_FELZENSZWALB_H
