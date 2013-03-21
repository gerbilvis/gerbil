#ifndef FULLIMAGESWITCHER_H
#define FULLIMAGESWITCHER_H

#include <background_task.h>
#include <multi_img.h>
#include <shared_data.h>


class FullImageSwitcher : public BackgroundTask {
public:
	enum SwitchTarget { REGULAR, LIMITED };
	FullImageSwitcher(multi_img_base_ptr limited, multi_img_ptr regular, SwitchTarget target)
		: limited(limited), regular(regular), target(target) {}
	virtual ~FullImageSwitcher() {}
	virtual bool run();
protected:
	multi_img_base_ptr limited;
	multi_img_ptr regular;
	SwitchTarget target;
};

#endif // FULLIMAGESWITCHER_H
