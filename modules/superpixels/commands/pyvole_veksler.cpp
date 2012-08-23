#include "veksler/veksler_segmentation.h"
#include "superpixel.h"

#include <boost/python.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pyvole {

  using namespace superpixels;

  cv::Mat veks(cv::Mat image, int patch_size = 20, int type = 0, int iterations = 2, int lambda = 10){
    VekslerSegmentation veksler(patch_size,type,iterations,lambda);
    cv::Mat_<cv::Vec3b>  Mat(image);
    std::vector<Superpixel> supixels = veksler.superpixels(Mat);
    return veksler.superpixelsImage(supixels,image.rows,image.cols,false);
    //    return true;
  }
    
  BOOST_PYTHON_FUNCTION_OVERLOADS(veks_overloads, veks, 1, 5);

  // Wrappers for overloaded functions
  std::vector<Superpixel> (VekslerSegmentation::*veksp1)(const cv::Mat_<cv::Vec3b>&) = &VekslerSegmentation::superpixels;
  cv::Mat_<cv::Vec3b> (VekslerSegmentation::*vekspI)(const std::vector<Superpixel>&, int, int, bool) = &VekslerSegmentation::superpixelsImage;
    
  BOOST_PYTHON_MODULE(_veksler){
    using namespace boost::python;

    def("segment_veksler",veks,veks_overloads(
					      args("image", "patch_size", "type", "iterations", "lambda"),
					      "Superpixel Segmentation by Veksler"
					      ));

    class_<superpixels::VekslerSegmentation>("VekslerSegmentation",init<int,int,int,int>())
      .def("superpixels", veksp1)
      .def("superpixelsImage", vekspI)
      ;
      
  }

}
