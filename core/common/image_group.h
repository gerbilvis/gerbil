#ifndef IMAGE_GROUP_H
#define IMAGE_GROUP_H

#include <fstream>
#include "global_defines.h"

class ImageGroup {
public:
	ImageGroup() { }
	ImageGroup(std::string original, std::string mask, std::string proc, std::string output, TStrSet tags) :
		original(original), mask(mask), proc(proc), output(output), tags(tags) { }

	void setOriginal(std::string original) { this->original = original; }
	std::string &getOriginal() { return original; }

	void setMask(std::string mask) { this->mask = mask; }
	std::string &getMask() { return mask; }

	void setProc(std::string proc) { this->proc = proc; }
	std::string &getProc() { return proc; }

	void setOutput(std::string output) { this->output = output; }
	std::string &getOutput() { return output; }

	void addTag(std::string tag) { tags.insert(tag); }
	void removeTag(std::string tag) { tags.erase(tag); }
	
	bool isSet(std::string tag) {
		return (tags.find(tag) != tags.end());
	}

	void setTags(TStrSet &tags) {
		this->tags = tags;
	}

	bool operator==(ImageGroup &ig) {
		return ((ig.getOriginal() == original) && 
			(ig.getMask() == mask) &&
			(ig.getProc() == proc) &&
			(ig.getOutput() == output));
	}

protected:
	std::string original;
	std::string mask;
	std::string proc;
	std::string output;
	TStrSet tags;
};

#endif // IMAGE_GROUP_H
