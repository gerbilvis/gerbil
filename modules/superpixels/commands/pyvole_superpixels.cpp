#include "superpixel.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace pyvole {

  using namespace superpixels;

  /*
    vector indexing suite for vectors with elements without == operator
  */

  template <class T> class no_compare_indexing_suite :  
    public boost::python::vector_indexing_suite<T, false, no_compare_indexing_suite<T> > 
  { 
  public: 
    static bool contains(T &container, typename T::value_type const &key) 
    { 
      PyErr_SetString(PyExc_NotImplementedError, "containment checking not supported on this container");
      throw boost::python::error_already_set();
    } 
  }; 

  BOOST_PYTHON_MODULE(_sp){
    using namespace boost::python;

    class_<std::vector<Superpixel> >("SuperpixelVec")
      .def(no_compare_indexing_suite<std::vector<Superpixel> >() );

    class_<std::vector<cv::Point> >("PointVec")
      .def(vector_indexing_suite<std::vector<cv::Point> >());

    class_<superpixels::Superpixel>("Superpixel")
      .def("empty",&superpixels::Superpixel::empty)
      .add_property("size",make_getter(&superpixels::Superpixel::size),make_setter(&superpixels::Superpixel::size))
      .add_property("coordinates",make_getter(&superpixels::Superpixel::coordinates))
      .add_property("bbox",make_getter(&superpixels::Superpixel::bbox))
      ;

  }

}
