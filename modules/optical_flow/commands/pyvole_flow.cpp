#include "flow.hpp"

#include <boost/python.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pyvole {

  cv::Mat3b vis_flow_mid(const cv::Mat_<float>& u, const cv::Mat_<float>& v, float maxflow = -1){
    cv::Mat3b image(u.size());
    if (maxflow>0) vision::middleburyDenseFlow<float>(image, u, v, maxflow);
    else vision::middleburyDenseFlow<float>(image, u, v);
    return image;
  }

  BOOST_PYTHON_FUNCTION_OVERLOADS(vis_mid_overloads, vis_flow_mid,2,3);

  BOOST_PYTHON_MODULE(_flow){
    using namespace boost::python;

    // expose a function: def("name in python",function,(optional) "user defined doc_string")
    def("visualize_middlebury",vis_flow_mid, vis_mid_overloads( args("u", "v", "maxflow"), "Middlebury visualization of dense optical flow\n\narg1 - u\n arg2 - v"));
      
  }

}