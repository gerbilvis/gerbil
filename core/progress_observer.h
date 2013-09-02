#ifndef PROGRESS_OBSERVER_H
#define PROGRESS_OBSERVER_H

namespace vole {

	class ProgressObserver
	{
	public:
		/// if false: cancel job
		virtual bool update(int percent) = 0;
	};
}

#endif // PROGRESS_OBSERVER_H
