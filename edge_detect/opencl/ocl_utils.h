#ifndef OCLUTILS_H
#define OCLUTILS_H

#include <CL/cl.hpp>
#include <string>

void print_ocl_err(cl::Error error);
void init_opencl(cl::Context& context, cl::CommandQueue& queue);

cl::Program build_cl_program(cl::Context& context,
                             const std::string& source_code,
                             const std::string& params = "");

std::string read_source(const std::string& path);

inline int round_up(const int value, const int multiplicity)
{
    return ((value + multiplicity - 1)/multiplicity) * multiplicity;
}

inline int round_up_power2(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}



#endif
