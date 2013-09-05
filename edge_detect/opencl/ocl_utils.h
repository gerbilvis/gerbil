#ifndef OCLUTILS_H
#define OCLUTILS_H

#include <CL/cl.hpp>
#include <string>

void print_ocl_err(cl::Error error);
void init_opencl(cl::Context& context, cl::CommandQueue& queue);

cl::Program build_cl_program(cl::Context& context,
                             const std::string& source_code);

std::string read_source(const std::string& path);

#endif
