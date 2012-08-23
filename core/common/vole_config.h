/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>
	and Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_CONFIG_H
#define VOLE_CONFIG_H

#ifdef WITH_BOOST_PROGRAM_OPTIONS
#include <boost/version.hpp>
#include <boost/program_options.hpp>
#endif // WITH_BOOST_PROGRAM_OPTIONS

#include "hashes.h"
#include <iostream>
#include <string>

namespace vole {

	/* base class that exposes configuration handling */
	class Config {
	public:
		Config(const std::string &prefix = std::string());

		virtual ~Config() {}

		virtual bool readConfig(const char *filename);

		bool storeConfig(const char *filename);
		virtual std::string getString() const = 0;

		virtual unsigned long int configHash(vole::HashMethod method = vole::HASH_djb2);

		#ifdef WITH_BOOST_PROGRAM_OPTIONS
			// takes a variables_map as optional argument, because there may be already one in use
			bool parseOptionsDescription(const char *filename,
				boost::program_options::variables_map *vm = NULL);

			boost::program_options::options_description options;
		#endif // WITH_BOOST_PROGRAM_OPTIONS

		/// helper function to be used in initBoostOptions
		const char* key(const char *key) const;

		/// verbosity level: 0 = silent, 1 = normal, 2 = a lot, 3 = insane
		int verbosity;
		/// cache for faster operation (declare this _before_ prefix!)
		bool prefix_enabled;
		/// config option prefix (may be empty)
		std::string prefix;

	protected:

		#ifdef WITH_BOOST_PROGRAM_OPTIONS
			virtual void initMandatoryBoostOptions();
		#endif // WITH_BOOST_PROGRAM_OPTIONS
	
	};
}


/* this is some macro trickery (just leave it as is) to make ENUMs
   work for reading (program_options) and writing (to stream) */
#ifdef WITH_BOOST_PROGRAM_OPTIONS
#include "boost/version.hpp"
#if BOOST_VERSION < 104200
#define ENUM_MAGIC(ENUM) \
	const char* ENUM ## Str[] = ENUM ## String;\
	void validate(boost::any& v, const std::vector<std::string>& values, \
	               ENUM* target_type, int) \
	{ \
		validators::check_first_occurrence(v); \
		const std::string& s = validators::get_single_string(values); \
		for (unsigned int i = 0; i < sizeof(ENUM ## Str)/sizeof(char*); ++i) { \
			if (strcmp(ENUM ## Str[i], s.c_str()) == 0) { \
				v = boost::any((ENUM)i); \
				return; \
			} \
		} \
		throw validation_error("invalid value"); \
	} \
	std::ostream& operator<<(std::ostream& o, ENUM e)  \
	{	o << ENUM ## Str[e]; return o;  }
#else	// only the exception throw is changed
#define ENUM_MAGIC(ENUM) \
	const char* ENUM ## Str[] = ENUM ## String;\
	void validate(boost::any& v, const std::vector<std::string>& values, \
	               ENUM* target_type, int) \
	{ \
		validators::check_first_occurrence(v); \
		const std::string& s = validators::get_single_string(values); \
		for (unsigned int i = 0; i < sizeof(ENUM ## Str)/sizeof(char*); ++i) { \
			if (strcmp(ENUM ## Str[i], s.c_str()) == 0) { \
				v = boost::any((ENUM)i); \
				return; \
			} \
		} \
		throw validation_error(validation_error::invalid_option_value); \
	} \
	std::ostream& operator<<(std::ostream& o, ENUM e)  \
	{	o << ENUM ## Str[e]; return o;  }\
//	
//	Note: the method below was a poor attempt to get ENUM_MAGIC to work (see
//	forensics/cmfd for a working example) for other modules as well.
//	However, it did not improve anything
/*
	std::istream& operator>>(std::istream& o, ENUM &e)  \
	{	std::string tmp;\
	    o >> tmp;\
		for (unsigned int i = 0; i < sizeof(ENUM ## Str)/sizeof(char*); ++i) {\
			if (strcmp(ENUM ## Str[i], tmp.c_str())) {\
				e = (ENUM)i; \
				return o;\
			}\
		}\
	}
*/
#endif // BOOST_VERSION
#else  // Now without Boost
#define ENUM_MAGIC(ENUM) \
	const char* ENUM ## Str[] = ENUM ## String;\
	std::ostream& operator<<(std::ostream& o, ENUM e)  \
	{	o << ENUM ## Str[e]; return o;  }
#endif // WITH_BOOST_PROGRAM_OPTIONS

#endif // VOLE_CONFIG_H
