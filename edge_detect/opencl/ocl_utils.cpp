#include "ocl_utils.h"

#include <iostream>
#include <fstream>

void print_ocl_err(cl::Error error)
{
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
}

void init_opencl(cl::Context& context, cl::CommandQueue& queue, bool profiling)
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

        // Select the default platform and create
        // a context using this platform and the GPU
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[0])(),
            0
        };
        context = cl::Context(CL_DEVICE_TYPE_ALL, cps);

        // Get a list of devices on this platform
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::cout << "num of devices: " << devices.size() << std::endl;

        std::cout << "device name: " << devices[0].getInfo<CL_DEVICE_NAME>()
                     << std::endl;

        // Create a command queue and use the first device

        cl_command_queue_properties props = 0;

        if(profiling)
            props |= CL_QUEUE_PROFILING_ENABLE;

        queue = cl::CommandQueue(context, devices[0], props);

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

void get_profile_info(cl::Event &event, cl_ulong &total_time)
{
    cl_ulong start = event
            .getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = event
            .getProfilingInfo<CL_PROFILING_COMMAND_END>();

    total_time = end - start;
}

void get_profile_info(std::vector<cl::Event> &events,
                      float &total_time, int &num_of_valid_events)
{
    cl_ulong total_time_ulong = 0, event_time;
    int counter = 0;

    for(int i = 0; i < events.size(); ++i)
    {
        cl::Event& event = events[i];

        try {
            get_profile_info(event, event_time);

            total_time_ulong += event_time;
            counter++;
        }
        catch (...)
        {
            break;
        }
    }

    num_of_valid_events = counter;
    total_time = total_time_ulong / 1000000.f;
}
