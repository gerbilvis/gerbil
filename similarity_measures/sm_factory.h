#ifndef SM_FACTORY_H
#define SM_FACTORY_H

#include "sm_config.h"
#include "l_norm.h"
#include "modified_spectral_angle_similarity.h"
#include "spectral_information_divergence.h"
#include "sidsam.h"
#include "normalized_l2.h"

namespace vole {

template<typename T>
class SMFactory {

public:
	/* factory method to spawn a specific similarity measure object */
	static SimilarityMeasure<T> *spawn(const SMConfig &c) {
		switch (c.measure) {
		case MANHATTAN:
			return new LNorm<T>(cv::NORM_L1);
		case EUCLIDEAN:
			return new LNorm<T>(cv::NORM_L2);
		case CHEBYSHEV:
			return new LNorm<T>(cv::NORM_INF);
		case MOD_SPEC_ANGLE:
			return new ModifiedSpectralAngleSimilarity<T>();
		case SPEC_INF_DIV:
			return new SpectralInformationDivergence<T>();
		case SIDSAM1:
			return new SIDSAM<T>(0);
		case SIDSAM2:
			return new SIDSAM<T>(1);
		case NORM_L2:
			return new NormalizedL2<T>();
		default:
			return 0;
		}
	}
};

}
#endif
