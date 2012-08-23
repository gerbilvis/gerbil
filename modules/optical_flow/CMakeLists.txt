vole_module_name("optical_flow")
vole_module_description("Algorithms for Optical Flow computation")
vole_module_variable("Vole_OpticalFlow")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS" "BOOST_SERIALIZATION" "BOOST_PYTHON" "PYTHON" "TBB")
vole_add_required_modules()

vole_compile_library(
	commands/command_hornschunck.cpp
	commands/command_hornschunck.h
	commands/hornschunck_config.cpp
	commands/hornschunck_config.h
	commands/command_pyramidalhornschunck.cpp
	commands/command_pyramidalhornschunck.h
	commands/pyramidalhornschunck_config.cpp
	commands/pyramidalhornschunck_config.h
)

vole_add_command("hornschunck" "commands/command_hornschunck.h" "vole::CommandHornschunck")
vole_add_command("pyramidal_hornschunck" "commands/command_pyramidalhornschunck.h" "vole::CommandPyramidalHornschunck")

# module names MUST BE EQUAL to the ones defined in the pyvole_bla.cpp
# BOOST_PYTHON_MODULE(module_name)

vole_add_python_module("_hornschunck" "commands/pyvole_hornschunck.cpp")
vole_add_python_module("_pyramidalhornschunck" "commands/pyvole_pyramidalhornschunck.cpp")
vole_add_python_module("_flow" "commands/pyvole_flow.cpp")

vole_compile_library(

  hornschunckopticalflow.hpp
  hornschunckopticalflow.cpp
  pyramidalhornschunckopticalflow.hpp
  pyramidalhornschunckopticalflow.cpp
  flow.hpp
  flow.cpp
  vision_config.hpp
  vision_export.hpp
  colorconversion.hpp
  colorconversion.cpp
  utility.hpp
  utility.cpp
)

vole_add_module()
