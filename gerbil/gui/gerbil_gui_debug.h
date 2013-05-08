#ifndef GERBIL_GUI_DEBUG_H
#define GERBIL_GUI_DEBUG_H

inline void YOU_FORGOT_TO_INCLUDE_IOSTREAMS() {
	std::cerr << "testing for std::cerr" << std::endl;
}

//#define DEBUG_FUN \
//		std::cerr << __FILE__ << __LINE__ << __func__ << std::endl;

#define DEBUG_FUN \
		std::cerr << __PRETTY_FUNCTION__ << std::endl;


#endif // GERBIL_GUI_DEBUG_H
