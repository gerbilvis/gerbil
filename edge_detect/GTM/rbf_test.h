#ifndef RBF_TEST_H
#define RBF_TEST_H

#include "highgui.h"

#include "rbfnet.h" 
#include "gtm.h" 
#include "gtm_em.h"

#include "edge_detection_config.h"



class RBF_Test
{
  public:
    RBF_Test(const EdgeDetectionConfig *conf);
    ~RBF_Test();
    
    void execute();

	private:
		const EdgeDetectionConfig *m_conf;
};

#endif