#ifndef IREAD_IMG_READ_CONFIG_H
#define IREAD_IMG_READ_CONFIG_H

#include "vole_config.h"

namespace iread {

	class IReadConfig : public vole::Config {
		public:
			IReadConfig(const std::string& prefix = std::string());
		public:
			virtual std::string getString() const;

		protected:
			#ifdef WITH_BOOST
				virtual void initBoostOptions();
			#endif // WITH_BOOST

		public:
			std::string imageFile;

			int dark_level;
			int saturation_level;
			std::string color_space;

		public:
	};
}

#endif //  IREAD_IMG_READ_CONFIG_H
