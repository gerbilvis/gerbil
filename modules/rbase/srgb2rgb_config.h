#ifndef RBASE_SRGB_TO_RGB_CONFIG_H
#define RBASE_SRGB_TO_RGB_CONFIG_H

#include "vole_config.h"
#include "img_read_config.h"

namespace rbase {

	class Srgb2RgbConfig : public vole::Config {
		public:
			Srgb2RgbConfig(const std::string& prefix = std::string());
		public:
			virtual std::string getString() const;

		protected:
			#ifdef WITH_BOOST
				virtual void initBoostOptions();
			#endif // WITH_BOOST

		public:
			iread::IReadConfig img;
			std::string output_file;
			int maxTargetIntensity;
	};
}

#endif //  RBASE_SRGB_TO_RGB_CONFIG_H
