vole_module_name("som")
vole_module_description("Self-organizing Map")
vole_module_variable("Gerbil_SOM")

vole_add_required_dependencies(OPENCV TBB BOOST)
vole_add_optional_dependencies(BOOST_PROGRAM_OPTIONS)
vole_add_required_modules(imginput similarity_measures)

# Command for simple testing, do not expose by default
# vole_add_command("somtest" "som_test.h" "som::SOMTest")

vole_compile_library(
	som_neuron.h
	som_config

	gensom

	isosom
	isosom_base
	isosom_2d
	isosom_3d
	isosom_4d

	som_cache
	som_distance

	#	som_test
)

vole_add_module()
