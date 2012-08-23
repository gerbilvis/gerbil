#ifndef VOLEEXCEPTION_H
#define VOLEEXCEPTION_H

#include <iostream>

class VoleException {
public:	
	VoleException() : What("vole must die (<- someone did not include a proper message here") {}
	VoleException(std::string what) : What(what) {}
	virtual ~VoleException() { }
	std::string what(void) 
	{ 
		return What; 
	} 

private:
	std::string What;
};

#endif // VOLEEXCEPTION_H
