#############################################################################
# enable_sse2()         Simple cmake macro to enable SSE2 SIMD extensions.
#############################################################################
#
# enable_sse2() does not test for sse, it simply sets the compiler switch
# to generate sse2 simd instructions.
#
# Supported compilers: MSVC, GCC

macro(enable_sse2)
	if(MSVC) # microsoft compiler
		if(CMAKE_CL_64)
		else()
			add_definitions("/arch:SSE2")
		endif()
	else(MSVC) # assume gcc
		add_definitions("-msse2")
	endif(MSVC)
endmacro()
