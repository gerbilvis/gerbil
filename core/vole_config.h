/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>
	and Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VOLE_CONFIG_H
#define VOLE_CONFIG_H

// Workaround for Qt 4 moc bug: moc fails to parse boost headers.
// Fixed in Qt 5.
// https://bugreports.qt-project.org/browse/QTBUG-22829
#ifndef Q_MOC_RUN
#ifdef WITH_BOOST_PROGRAM_OPTIONS
#include <boost/version.hpp>
#include <boost/program_options.hpp>
#endif // WITH_BOOST_PROGRAM_OPTIONS
#endif // Q_MOC_RUN

#include "hashes.h"
#include <iostream>
#include <string>

/* some helpful macros */
#ifdef WITH_BOOST
#define DESC_OPT(opt, s) const char * opt = s;
#define BOOST_OPT(opt) (\
	key(#opt), \
	boost::program_options::value(&opt)->default_value(opt), \
	desc::opt)
// option with one digit shortcut (shopt)
#define BOOST_OPT_S(opt,shopt) (\
	key(#opt","#shopt), \
	boost::program_options::value(&opt)->default_value(opt), \
	desc::opt)
#define BOOST_BOOL(opt) (\
	key(#opt), \
	boost::program_options::bool_switch(&opt)->default_value(opt), \
	desc::opt)
#endif
#define COMMENT_OPT(s, opt) s << #opt "=" << opt  \
	<< " # " <<  desc::opt << std::endl
#define COMMENT_BOOL(s, opt) s << #opt "=" << (opt ? "true" : "false" )  \
	<< " # " <<  desc::opt << std::endl

/* base class that exposes configuration handling */
class Config {
public:
	Config(const std::string &prefix = std::string());
	Config(const Config &other);
	/* NOTE: assignment does not include the options member! */
	Config& operator=(const Config& other);

	virtual ~Config() {}

	virtual bool readConfig(const char *filename);

	bool storeConfig(const char *filename);
	virtual std::string getString() const = 0;

	virtual unsigned long int
	configHash(Hashes::Method method = Hashes::HASH_djb2);

#ifdef WITH_BOOST_PROGRAM_OPTIONS
	// takes a variables_map as optional argument, because there may be already one in use
	bool parseOptionsDescription(const char *filename,
								 boost::program_options::variables_map *vm = NULL);
#endif // WITH_BOOST_PROGRAM_OPTIONS

	/// helper function to be used in initBoostOptions
	const char* key(const char *key) const;

	/// verbosity level: 0 = silent, 1 = normal, 2 = a lot, 3 = insane
	int verbosity;
	/// cache for faster operation (declare this _before_ prefix!)
	bool prefix_enabled;
	/// config option prefix (may be empty)
	std::string prefix;

#ifdef WITH_BOOST_PROGRAM_OPTIONS
	boost::program_options::options_description options;
#endif // WITH_BOOST_PROGRAM_OPTIONS

protected:

#ifdef WITH_BOOST_PROGRAM_OPTIONS
	virtual void initMandatoryBoostOptions();
#endif // WITH_BOOST_PROGRAM_OPTIONS

};


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
#define ENUM_MAGIC(NAMESPACE, ENUM) \
	const char* ENUM ## Str[] = NAMESPACE ## _ ## ENUM ## String;\
	void validate(boost::any& v, const std::vector<std::string>& values, \
	               ENUM* target_type, int) \
	{ \
		using namespace boost::program_options; \
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
	{	o << ENUM ## Str[e]; return o;  }

// For enums defined inside class scope.
#define ENUM_MAGIC_CLS(CLAZZ, ENUM) \
	const char* CLAZZ ## _ ## ENUM ## _ ## Str[] = \
		CLAZZ ## _ ## ENUM ## _ ## String;\
	void validate(boost::any& v, const std::vector<std::string>& values, \
				  CLAZZ::ENUM* target_type, int) \
	{ \
		using namespace boost::program_options; \
		validators::check_first_occurrence(v); \
		const std::string& s = \
			validators::get_single_string(values); \
		for (unsigned int i = 0; \
			 i < sizeof(CLAZZ ## _ ## ENUM ## _ ## Str)/sizeof(char*);\
			 ++i) \
		{ \
			if (strcmp(CLAZZ ## _ ## ENUM ## _ ## Str[i], s.c_str()) == 0) { \
				v = boost::any((CLAZZ::ENUM)i); \
				return; \
			} \
		} \
		throw validation_error(validation_error::invalid_option_value); \
	} \
	std::ostream& operator<<(std::ostream& o, CLAZZ::ENUM e)  \
	{	o << CLAZZ ## _ ## ENUM ## _ ## Str[e]; return o;  }

#endif // BOOST_VERSION
#else  // Now without Boost
#define ENUM_MAGIC(ENUM) \
	const char* ENUM ## Str[] = ENUM ## String;\
	std::ostream& operator<<(std::ostream& o, ENUM e)  \
	{	o << ENUM ## Str[e]; return o;  }
#endif // WITH_BOOST_PROGRAM_OPTIONS

#endif // VOLE_CONFIG_H
