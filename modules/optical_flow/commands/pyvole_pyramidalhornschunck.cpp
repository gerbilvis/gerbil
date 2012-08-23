#include "pyramidalhornschunckopticalflow.hpp"

#include <boost/python.hpp>
#include <boost/python/overloads.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pyvole {

  boost::python::tuple ofhsp(cv::Mat1b image1, cv::Mat1b image2, float alpha = 5, unsigned iterations = 100, unsigned levels = 5, float scale = 2) {
    vision::PyramidalHornSchunckOpticalFlow<float> hs;
    hs.setAlpha(alpha);
    hs.setIterations(iterations);
    hs.setLevels(levels);
    hs.setScaleFactor(scale);

    cv::Mat_<float> u, v;
    hs.compute(image1, image2, u, v);
    return boost::python::make_tuple(u, v);
  }

  BOOST_PYTHON_FUNCTION_OVERLOADS(ofhsp_overloads,ofhsp,2,6);

  BOOST_PYTHON_MODULE(_pyramidalhornschunck){
    using namespace boost::python;

    // expose a function: def("name in python",function,(optional) "user defined doc_string")
    def("horn_schunck_pyramidal",ofhsp, ofhsp_overloads(args("previous", "current", "alpha", "iterations", "levels", "scaleFactor")));

	//, "arg1 - previous img\n arg2 - current img\n arg3 - alpha\n arg4 - iterations\n arg5 - levels\n scale factor");
      
  }

}
