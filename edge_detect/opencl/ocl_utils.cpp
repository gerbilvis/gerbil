#include "ocl_utils.h"

#include <iostream>
#include <fstream>

void print_ocl_err(cl::Error error)
{
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
}

void init_opencl(cl::Context& context, cl::CommandQueue& queue)
{
    try
    {
        // Get available platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (!platforms.size())
        {
            std::cout << "Platform size 0" << std::endl;
        }
        else
        {
            std::cout << "Platforms size: " << platforms.size() << std::endl;
            std::string platform_name = platforms[0].getInfo<CL_PLATFORM_NAME>();

            std::cout << "Platform name: " << platform_name << std::endl;
        }

        // Select the default platform and create a context using this platform and the GPU
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[0])(),
            0
        };
        context = cl::Context(CL_DEVICE_TYPE_ALL, cps);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        queue = cl::CommandQueue(context, devices[0]);

    }
    catch (cl::Error err)
    {
        print_ocl_err(err);
    }

}

cl::Program build_cl_program(cl::Context& context,
                             const std::string& source_code,
                             const std::string& params)
{
    cl::Program::Sources source(1, std::make_pair(source_code.c_str(),
                                                  source_code.length()+1));

    // Make program of the source code in the context
    cl::Program program = cl::Program(context, source);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Build program for these specific devices
    try
    {
        program.build(devices, params.c_str());
    }
    catch(cl::Error ex)
    {
        std::cout << std::endl;
        std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        throw ex;
    }

    return program;
}

std::string read_source(const std::string& path)
{
    std::ifstream sourceFile(path.c_str());
    std::string sourceCode(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));

    return sourceCode;
}
