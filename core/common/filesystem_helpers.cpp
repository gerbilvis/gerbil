/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#include "filesystem_helpers.h"
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include "string.h"

bool FilesystemHelpers::file_exists(std::string filename)
{
	struct stat stFileInfo;
	int intStat;

	// Attempt to get the file attributes
	intStat = stat(filename.c_str(),&stFileInfo);
	if(intStat == 0) {
		// We were able to get the file attributes
		// so the file obviously exists.
	return true;
	} else {
		// We were not able to get the file attributes.
		// This may mean that we don't have permission to
		// access the folder which contains this file. If you
		// need to do that level of checking, lookup the
		// return values of stat which will give you
		// more details on why stat failed.
		perror("stat");
/*
       EACCES Search permission is denied for one of the directories in the path prefix of path.
				(See also path_resolution(7).)
       EBADF  fd is bad.
       EFAULT Bad address.
       ELOOP  Too many symbolic links encountered while traversing the path.
       ENAMETOOLONG File name too long.
       ENOENT A component of the path path does not exist, or the path is an empty string.
       ENOMEM Out of memory (i.e., kernel memory).
       ENOTDIR A component of the path is not a directory.
 */

	return false;
	}
}


std::string FilesystemHelpers::strip_last_filename_component(std::string file)
{
	std::string workfile = file;
	size_t pos;
	unsigned int file_length;
	do {
		file_length = workfile.length();
		pos = workfile.find_last_of('/');
		if (pos == std::string::npos) { // if stripped, the result will be 0.
			return "";
		}
		if (pos >= file_length) {  // ok, it ends with a slash, so let's trunkate it.
			workfile.resize(file_length - 1);
		} else {
			// found one (or more) non-slash end character - this should be removed
			workfile.resize(pos); // this is the chopping we need.
			return workfile; // done.
		}
	} while (pos >= file_length);
	// just in case...
	return "";
}


bool FilesystemHelpers::recursive_mkdir(std::string path)
{
// FIXME
#ifdef _GNUCPP
	int mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH; // file mask 775
	if (path.length() == 0) return true;
	// try to mkdir path
	if (mkdir(path.c_str(), mode) != 0) { // fail? 
		if (errno == EEXIST) return true; // dir exists already
		if (errno == ENOENT) {
			if (recursive_mkdir(strip_last_filename_component(path)) == false) {
				strerror(errno);
				return false; // was it possible to create higher-level directories?
			}
			if (mkdir(path.c_str(), mode) == 0) {
				return true; // yes, directories fixed
			} else {
				strerror(errno);
				return false; // something unrecoverable
			}
		}
		strerror(errno);
		if (errno == EROFS) return false; // read-only filesystem :(
		if (errno == EMLINK) return false; // link count exceeded in parent directory
		if (errno == EACCES) return false; // permission problem: check rwx of parent dirs
		if (errno == ENOTDIR) return false; // something in the path is no directory
		if (errno == ENAMETOOLONG) return false; // a path component is too long
		if (errno == ELOOP) return false; // loop in directory names
	}
#endif //_GNUCPP
	return true;
}

// TODO function looks buggy; better test it
std::string FilesystemHelpers::basename_without_extension(std::string file)
{
	size_t pos = file.find_last_of('/');
	if (pos == std::string::npos) {
		pos = 0;
	} else {
		pos++;
	}
	if (pos >= file.length()) { // appears to have a trailing slash, error 
		return "";
	}
	size_t pos2 = file.find_last_of('.');
	if (pos2 < pos) pos2 = std::string::npos;
	return file.substr(pos, pos2 - pos);
}


