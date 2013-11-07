vole_module_name("RBC")
vole_module_description("RBC OpenCL implementation based on works from: http://www.lcayton.com/code.html")
vole_module_variable("Gerbil_RBC")

vole_add_required_dependencies("OPENCV")
vole_add_required_dependencies("OpenCL")
vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")

vole_compile_library(
        "RBC"
)

vole_add_command("RBC" "RBC.h" "RBC")


# loading kernels
file(GLOB cl_kernels "opencl/kernels/*.cl")

set(ocl_kernels_file opencl/kernels.cpp)

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}"
    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}" -P "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake"
    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake")


vole_compile_library("RBC"
                     "RBC_config"
                     "driver"
                     "brute"
#                    "kernels"
                     "kernelWrap"
                     "rbc_ops"
#                    "sKernel"
                     "sKernelWrap"
                     "utils"
                     "utilsGPU"
                     "opencl/kernels/rbc.cl"
                     "opencl/kernels")
vole_add_module()


##vole_module_name("rbc")
##vole_module_description("Random Ball Cover OpenCL implementation")
##vole_module_variable("Gerbil_RBC")

##vole_add_required_dependencies("OpenCL")
##vole_add_optional_dependencies("BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_FILESYSTEM")

##vole_compile_library(
##        "rbc"
##)
##find_package(OpenCL REQUIRED)


## loading kernels
#file(GLOB cl_kernels "opencl/kernels/*.cl")

#set(ocl_kernels_file opencl/kernels.cpp)

#add_custom_command(
#    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}"
#    COMMAND ${CMAKE_COMMAND} -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/opencl/kernels" -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file}" -P "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake"
#    DEPENDS ${cl_kernels} "${CMAKE_SOURCE_DIR}/cmake/cl2cpp.cmake")

#add_executable("rbc" "RBC" "driver"
#                        "brute"
##                        "kernels"
#                        "kernelWrap"
#                        "rbc_ops"
##                        "sKernel"
#                        "sKernelWrap"
#                        "utils"
#                        "utilsGPU"
#                        "opencl/kernels/rbc.cl"
#                        "opencl/kernels")



#target_link_libraries("rbc" ${OPENCL_LIBRARY})

##vole_compile_library(
##        "main"
##)

##vole_add_module()
