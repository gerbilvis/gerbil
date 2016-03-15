vole_module_name("mapper")
vole_module_description("Similarity Heatmap using Spectral Angle Mapper (SAM)")
vole_module_variable("Gerbil_Mapper")

vole_add_required_modules(similarity_measures imginput)
vole_add_required_dependencies(TBB)

vole_compile_library(
	mapper
	mapper_config
)

vole_add_command("mapper" "mapper.h" "mapper::Mapper")

vole_add_module()
