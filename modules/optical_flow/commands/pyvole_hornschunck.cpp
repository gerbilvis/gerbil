#include "hornschunckopticalflow.hpp"

#include <boost/python.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pyvole {

  boost::python::tuple ofhs(cv::Mat1b image1, cv::Mat1b image2, float alpha = 5, unsigned iterations = 100) {
    vision::HornSchunckOpticalFlow<float> hs;
    hs.setAlpha(alpha);
    hs.setIterations(iterations);
    cv::Mat_<float> u, v;
    hs.compute(image1, image2, u, v);
    return boost::python::make_tuple(u, v);
  }

  BOOST_PYTHON_FUNCTION_OVERLOADS(ofhs_overloads,ofhs,2,4);

  BOOST_PYTHON_MODULE(_hornschunck){
    using namespace boost::python;

    // expose a function: def("name in python",function,(optional) "user defined doc_string")
    def("horn_schunck",ofhs,ofhs_overloads(
					   args("previous img", "current img", "alpha", "iterations"),
					   "Calculate dense optical flow after Horn & Schunck"
					   ));
      
  }

}
