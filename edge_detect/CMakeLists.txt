vole_module_name("edge_detect")
vole_module_description("Edge detection in multispectral images")
vole_module_variable("Gerbil_Edge_Detect")

vole_add_required_dependencies("OPENCV")
vole_add_required_dependencies("OpenCL")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")
vole_add_required_modules("similarity_measures")

vole_compile_library(
        "edge_detection"
)

vole_add_command("edge_detect" "edge_detection.h" "EdgeDetection")

# loading kernels
file(GLOB cl_kernels "opencl/kernels/*.cl")

set(ocl_kernels_file opencl/kernels.cpp)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}"
    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}" -P "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake"
    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake")


vole_compile_library(
        "edge_detection"
        "edge_detection_config"
        "neuron"
        "som"
        "som2d"
        "som3d"
        "som_cone"
        "som_distance"
        "som_tester"
        "som_trainer"
        "opencl/ocl_som2d"
#        "opencl/ocl_som2d_new"
        "opencl/ocl_som2d_cpu_opt"
        "opencl/ocl_som3d"
        "opencl/ocl_utils"
        "opencl/ocl_som_types"
        "opencl/kernels" #generated
        "opencl/kernels/som2d.cl"
        "opencl/kernels/som2d_cpu_opt.cl"
        "opencl/kernels/som3d.cl"
)

vole_add_module()


#file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/kernels)

## it should be automated using list and loop
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels/vector_add.cl
#    ${PROJECT_BINARY_DIR}/bin/kernels/vector_add.cl COPYONLY)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels/som2d.cl
#    ${PROJECT_BINARY_DIR}/bin/kernels/som2d.cl COPYONLY)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels/som2d_new.cl
#    ${PROJECT_BINARY_DIR}/bin/kernels/som2d_new.cl COPYONLY)

#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels/som2d_cpu_opt.cl
#    ${PROJECT_BINARY_DIR}/bin/kernels/som2d_cpu_opt.cl COPYONLY)
