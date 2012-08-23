#ifndef FILESYSTEM_HELPERS_H
#define FILESYSTEM_HELPERS_H

#include <sys/stat.h>
#include <iostream>

class FilesystemHelpers {
public:

	static bool file_exists(std::string filename);

	/* FIXME double-check: function is maybe not portable */
	static std::string strip_last_filename_component(std::string file);

	static bool recursive_mkdir(std::string path);

	// TODO function looks buggy; better test it
	static std::string basename_without_extension(std::string file);

};




#endif // FILESYSTEM_HELPERS_H
