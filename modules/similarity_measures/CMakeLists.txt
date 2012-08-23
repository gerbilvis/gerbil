vole_module_name("similarity_measures")
vole_module_description("Distance measures for grayscale images")
vole_module_variable("Vole_Similarity_Measures")

vole_add_required_dependencies("OPENCV" "BOOST" "BOOST_PROGRAM_OPTIONS")

vole_compile_library(
	"command_similarity_measure"
)

vole_add_command("dist" "command_similarity_measure.h" "vole::CommandSimilarityMeasure")

vole_compile_library(
	"correlation_coefficient_histogram"
	"earth_movers_distance"
	"gradient_difference"
	"mean_squares"
	"mean_squares_histogram"
	"mean_reciprocal_square_difference"
	"mutual_information_histogram"
	"normalized_cross_correlation"
	"normalized_mutual_information"
	"similarity_measure"
	"l_norm"
	"modified_spectral_angle_similarity"
	"spectral_information_divergence"
	"sidsam"
	"normalized_l2"

	"sm_config" "sm_factory"
	"similarity_measure_core"
	"similarity_measure_config"
)

vole_add_module()
