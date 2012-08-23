/*
  pyvole_common.cpp
  author: David Bernecker (david.bernecker@cs.fau.de)

  Contains useful boost::python extensions for pyvole:
  - converters for ndarray <-> cv::Mat_<T>
  - converters for ndarray <-> cv::Point
*/


#ifndef VOLE_COMMON_PY_EXTENSION_H
#define VOLE_COMMON_PY_EXTENSION_H

#include <opencv2/core/core.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <Python.h>
#include <numpy/arrayobject.h>

#define CV_DEPTH_TO_NP_TYPE_RETPYNONE(CVDEPTH,NP_TYPE)			\
  switch (CVDEPTH){							\
  case 0: NP_TYPE = PyArray_UINT8; break;				\
  case 1: NP_TYPE = PyArray_INT8; break;				\
  case 2: NP_TYPE = PyArray_UINT16; break;				\
  case 3: NP_TYPE = PyArray_INT16; break;				\
  case 4: NP_TYPE = PyArray_INT32; break;				\
  case 5: NP_TYPE = PyArray_FLOAT32; break;				\
  case 6: NP_TYPE = PyArray_FLOAT64; break;				\
  default: std::cerr<<"Mat object has unknown type!"<<std::endl;	\
    Py_RETURN_NONE;							\
  }									\

#define CV_DEPTH_TO_NP_TYPE(CVDEPTH,NP_TYPE)				\
  switch (CVDEPTH){							\
  case 0: NP_TYPE = PyArray_UINT8; break;				\
  case 1: NP_TYPE = PyArray_INT8; break;				\
  case 2: NP_TYPE = PyArray_UINT16; break;				\
  case 3: NP_TYPE = PyArray_INT16; break;				\
  case 4: NP_TYPE = PyArray_INT32; break;				\
  case 5: NP_TYPE = PyArray_FLOAT32; break;				\
  case 6: NP_TYPE = PyArray_FLOAT64; break;				\
  default: NP_TYPE = -1;						\
  }									\

namespace pyvole {

  /*
    cv::Point_<> -> ndarray
   */

  template <typename PointT> class Point_to_python_ndarray{
  public:
    static PyObject* convert(cv::Point_<PointT> pt){
      int depth = cv::DataType<cv::Point_<PointT> >::depth;
      int np_type;
      CV_DEPTH_TO_NP_TYPE_RETPYNONE(depth,np_type);
      Py_ssize_t shape = 2;
      PyObject* arr_obj = PyArray_SimpleNew(1,&shape,np_type);
      PointT * element = reinterpret_cast<PointT*>(PyArray_GETPTR1(arr_obj,0));
      (*element) = pt.x;
      element =  reinterpret_cast<PointT*>(PyArray_GETPTR1(arr_obj,1));
      (*element) = pt.y;
      return boost::python::incref(arr_obj);
    }
  };

  /*
    cv::Mat -> ndarray
  */

  struct Mat_to_python_ndarray{

    static PyObject * convert(cv::Mat mat){
      if (!mat.isContinuous()) {
	std::cerr<<"Mat object to convert is not continuous!"<<std::endl;
	Py_RETURN_NONE;
      }
      // get Mat dimensions and shape
      int nd = (mat.channels()>1) ? 3 : 2;
      Py_ssize_t shape[nd];
      shape[0]=mat.rows; shape[1]=mat.cols;
      if (nd==3) shape[2]=3;
      // get Mat type
      int np_type;
      CV_DEPTH_TO_NP_TYPE_RETPYNONE(mat.depth(),np_type);
      //construct ndarray from data
      PyObject * arr_obj = PyArray_SimpleNewFromData(nd,shape,np_type,(void*)mat.data);
      mat.addref();
      return boost::python::incref(arr_obj);
    }

  };

  /*
    cv::Mat_<T> -> ndarray
   */

  template <typename T>
  class Mat_T_to_python_ndarray{

  public:
    static PyObject * convert(cv::Mat_<T> mat){
      if(!mat.isContinuous()) {
	std::cerr<<"Mat object to convert is not continuous!"<<std::endl;
	Py_RETURN_NONE;
      }
      int nd = (mat.channels()>1) ? 3 : 2;
      Py_ssize_t shape[nd];
      shape[0]=mat.rows; shape[1]=mat.cols;
      if ( nd == 3 ) shape[2] = mat.channels();
      int np_type;
      CV_DEPTH_TO_NP_TYPE_RETPYNONE(mat.depth(),np_type);
      PyObject * arr_obj = PyArray_SimpleNewFromData(nd,shape,np_type,(void*)mat.data);
      mat.addref();
      return boost::python::incref(arr_obj);
    }

  };


  /*
    ndarray -> cv::Mat
  */

  struct Mat_from_python_ndarray{
    /*
      Register Converter with boost.python when a Mat_from_python_ndarray object is created
    */
    Mat_from_python_ndarray(){
      boost::python::converter::registry::push_back(&convertible,&construct,boost::python::type_id<cv::Mat>());
    }

    /*
      Check if type of object is a sub-type of PyArray_Type,
      if not, then do not convert.
    */
    static void* convertible(PyObject* obj_ptr){
      if (!PyArray_Check(obj_ptr)) return NULL;
      if (PyArray_NDIM(obj_ptr)<2) return NULL;
      return obj_ptr;
    }

    /*
      Convert the object to a cv::Mat
    */
    static void construct(PyObject* obj_ptr,
			  boost::python::converter::rvalue_from_python_stage1_data* data){
      // Get information about array, shape, strides, etc.
      int ndim = PyArray_NDIM(obj_ptr);
      Py_ssize_t * shape = PyArray_DIMS(obj_ptr);
      Py_ssize_t * strides = PyArray_STRIDES(obj_ptr);
      int itemsize = PyArray_ITEMSIZE(obj_ptr);
      size_t steps[2]; int size[2]; // ndim >=2 assured by check in convertible
      for (int i=0;i<2;++i){
	steps[i]=strides[i]; // PyArray strides are number of elements, not bytes
	size[i]=(int)shape[i];
      }
      void * arr_data = PyArray_DATA(obj_ptr);

      // Check datatype of array
      int cv_depth = -1;
      if (PyArray_ISUNSIGNED(obj_ptr)) {
	if (itemsize == 1)
	  cv_depth = 0;
	else if (itemsize == 2)
	  cv_depth = 2;
      }
      else if(PyArray_ISSIGNED(obj_ptr)) {
	if (itemsize == 8)
	  cv_depth = 1;
	else if (itemsize == 2)
	  cv_depth = 3;
	else if (itemsize == 3)
	  cv_depth = 4;
      }
      else if(PyArray_ISFLOAT(obj_ptr)) {
	if (itemsize == 3)
	  cv_depth = 5;
	else if (itemsize == 4)
	  cv_depth = 6;
      }
    
      if (ndim==3){
	cv_depth = CV_MAKE_TYPE(cv_depth,(int)shape[2]);
      }

      // Get pointer to where the cv::Mat should be constructed
      void * storage = ((boost::python::converter::rvalue_from_python_storage<cv::Mat> *) data)->storage.bytes;

      // Construct cv::Mat
      new (storage) cv::Mat(2,size,cv_depth,arr_data,steps);

      data->convertible = storage;
    }

  };

  /*
    ndarray -> cv::Mat_<T>
  */

  template <typename T>
  class Mat_T_from_python_ndarray{
  public:
    /*
      Register converter when a Mat_T_from_python_ndarray object is created
     */
    Mat_T_from_python_ndarray(){
      boost::python::converter::registry::push_back(&convertible,
						    &construct,
						    boost::python::type_id<cv::Mat_<T> >());
    }

    /*
      Check if object is a sub-type of PyArray_Type,
      also check if type, numbers of dimensions and channels match.
    */
    static void* convertible(PyObject* obj_ptr){
      if (!PyArray_Check(obj_ptr)) return NULL;
      int channels = cv::DataType<T>::channels;
      if (channels > 1 && PyArray_NDIM(obj_ptr)!=3 )
	return NULL;
      if (PyArray_NDIM(obj_ptr)==3) {
	if (PyArray_DIMS(obj_ptr)[2] != channels) return NULL; // ndarray does not have right number of channels
      }	// number of channels matches
      PyArray_Descr* dtype_ptr = PyArray_DESCR(obj_ptr);
      int np_type = dtype_ptr->type_num;
      int cv_type;
      CV_DEPTH_TO_NP_TYPE(cv::DataType<T>::depth,cv_type);
      if (np_type != cv_type) return NULL;
      return obj_ptr;
    }

    static void construct(PyObject* obj_ptr,
			  boost::python::converter::rvalue_from_python_stage1_data* data){
      // Get information about array, shape, strides, etc.
      int ndim = PyArray_NDIM(obj_ptr);
      Py_ssize_t * shape = PyArray_DIMS(obj_ptr);
      Py_ssize_t * strides = PyArray_STRIDES(obj_ptr);
      size_t steps[2]; int size[2]; // ndim >=2 assured by check in convertible
      for (int i=0;i<2;++i){
	steps[i]=strides[i];
	size[i]=(int)shape[i];
      }
      void * arr_data = PyArray_DATA(obj_ptr);

      // Get pointer to where the cv::Mat should be constructed
      void * storage = ((boost::python::converter::rvalue_from_python_storage<cv::Mat_<T> > *) data)->storage.bytes;

      // Construct cv::Mat
      new (storage) cv::Mat_<T>(size[0],size[1],(T*) arr_data,steps[0]);

      data->convertible = storage;
    }
  };



  /*
    init all converters in a single function
  */

  void initializeConverters() {

    import_array();

    // cv::Mat -> ndarray
    boost::python::to_python_converter<cv::Mat,Mat_to_python_ndarray>();

    boost::python::to_python_converter<cv::Mat_<unsigned char>,Mat_T_to_python_ndarray<unsigned char> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2b>,Mat_T_to_python_ndarray<cv::Vec2b> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3b>,Mat_T_to_python_ndarray<cv::Vec3b> >();

    boost::python::to_python_converter<cv::Mat_<short>,Mat_T_to_python_ndarray<short> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2s>,Mat_T_to_python_ndarray<cv::Vec2s> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3s>,Mat_T_to_python_ndarray<cv::Vec3s> >();

    boost::python::to_python_converter<cv::Mat_<unsigned short>,Mat_T_to_python_ndarray<unsigned short> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2w>,Mat_T_to_python_ndarray<cv::Vec2w> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3w>,Mat_T_to_python_ndarray<cv::Vec3w> >();

    boost::python::to_python_converter<cv::Mat_<int>,Mat_T_to_python_ndarray<int> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2i>,Mat_T_to_python_ndarray<cv::Vec2i> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3i>,Mat_T_to_python_ndarray<cv::Vec3i> >();

    boost::python::to_python_converter<cv::Mat_<float>,Mat_T_to_python_ndarray<float> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2f>,Mat_T_to_python_ndarray<cv::Vec2f> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3f>,Mat_T_to_python_ndarray<cv::Vec3f> >();

    boost::python::to_python_converter<cv::Mat_<double>,Mat_T_to_python_ndarray<double> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec2d>,Mat_T_to_python_ndarray<cv::Vec2d> >();
    boost::python::to_python_converter<cv::Mat_<cv::Vec3d>,Mat_T_to_python_ndarray<cv::Vec3d> >();

    // ndarray -> cv::Mat
    Mat_from_python_ndarray();

    Mat_T_from_python_ndarray<unsigned char>();
    Mat_T_from_python_ndarray<cv::Vec2b>();
    Mat_T_from_python_ndarray<cv::Vec3b>();

    Mat_T_from_python_ndarray<short>();
    Mat_T_from_python_ndarray<cv::Vec2s>();
    Mat_T_from_python_ndarray<cv::Vec3s>();

    Mat_T_from_python_ndarray<unsigned short>();
    Mat_T_from_python_ndarray<cv::Vec2w>();
    Mat_T_from_python_ndarray<cv::Vec3w>();

    Mat_T_from_python_ndarray<int>();
    Mat_T_from_python_ndarray<cv::Vec2i>();
    Mat_T_from_python_ndarray<cv::Vec3i>();

    Mat_T_from_python_ndarray<float>();
    Mat_T_from_python_ndarray<cv::Vec2f>();
    Mat_T_from_python_ndarray<cv::Vec3f>();

    Mat_T_from_python_ndarray<double>();
    Mat_T_from_python_ndarray<cv::Vec2d>();
    Mat_T_from_python_ndarray<cv::Vec3d>();
  }

  void printPyvoleInfo(){
    std::cout<<"Vole meets Python..."<<std::endl;
  }

  BOOST_PYTHON_MODULE(_common){
    using namespace boost::python;

    initializeConverters();

    class_<cv::Rect>("Rect")
      .add_property("x",make_getter(&cv::Rect::x),make_setter(&cv::Rect::x))
      .add_property("y",make_getter(&cv::Rect::y),make_setter(&cv::Rect::y))
      .add_property("width",make_getter(&cv::Rect::width),make_setter(&cv::Rect::width))
      .add_property("height",make_getter(&cv::Rect::height),make_setter(&cv::Rect::height))
      ;

    class_<cv::Point>("Point")
      .add_property("x",make_getter(&cv::Point::x),make_setter(&cv::Point::x))
      .add_property("y",make_getter(&cv::Point::y),make_setter(&cv::Point::y))
      ;

    def("printInfo",printPyvoleInfo);

  }

}

#endif
