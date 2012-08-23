/*	
	Copyright(c) 2010 Christian Riess <christian.riess@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef IMAGE_LIST_H
#define IMAGE_LIST_H

#include "image_group.h"

class ImageList {
public:
	ImageList() { }
	ImageList(int size) { images.reserve(size); }

	void add(ImageGroup *ig) {
		images.push_back(ig);
	}
	void remove(ImageGroup *ig) {
		for (std::vector<ImageGroup *>::iterator it = images.begin(); it != images.end(); it++) {
			if ((*it) == ig) {
				images.erase(it);
				break;
			}
		}
	}

	std::vector<ImageGroup *> selectTag(std::string tag) {
		std::vector<ImageGroup *> selection;
		for (std::vector<ImageGroup *>::iterator it = images.begin(); it != images.end(); it++) {
			if ((*it)->isSet(tag)) {
				selection.push_back(static_cast<ImageGroup *>(*it));
			}
		}
		return selection;
	}

	size_t size() {
		return images.size();
	}

	ImageGroup *get(int i) {
		return images[i];
	}


protected:
	std::vector<ImageGroup *> images;
};

#endif // IMAGE_LIST_H
