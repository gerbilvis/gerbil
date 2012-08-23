/*	
	Copyright(c) 2011 Johannes Jordan <johannes.jordan@cs.fau.de>.

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef LABELING_H
#define LABELING_H

#ifdef WITH_OPENCV2 // theoretically, vole could be built w/o opencv..

#include <cv.h>
#include <string>
#include <sstream>
#include <iostream>

namespace vole {

/** Class for per-pixel label processing.

This class provides methods to import and export pixel masks or labels as
images. Another application is seeding input to supervised segmentation.

The main purpose of this code is to understand and provide "human-readable"
labeling images, where the color of a label can be chosen freely. All distinct
colors observed in the label image lead to a distinct entry in the label set.

The second purpose is to write out labeling images that are human-readable. You
can also let this class create you nice label colors for your own use.

@todo This is also the place for code to "mark" an image, e.g. overlay pixels
with a label color, or in the binary case, highlighting marked pixels. Code for
this already exists in forensics/cmfd.

**/
class Labeling {

public:

	/** Initialize without data.
		@arg labelcount Set labelcount to create label colors without data.
	**/
	Labeling(int labelcount = 0) : yellowcursor(true), labelcount(labelcount) {}

	/** Initialize with matrix containing arbitrary labels.
		See read().
		@note If your label matrix is a good citizen, consider operator=
		instead: Labeling l = m;
	**/
	Labeling(const cv::Mat &labeling, bool binary);
	Labeling(const std::string &filename, bool binary);
	virtual ~Labeling() {}

	/// Construct from label matrix. See setLabels.
	Labeling(const cv::Mat &labeling) : yellowcursor(true) {
		setLabels(labeling);
	}

	/// Set label matrix. See setLabels.
	Labeling & operator=(const cv::Mat &labeling) {
		setLabels(labeling);
		return *this;
	}

	/// Get label matrix, index 0 means "empty" label, indices are subsequent.
	const cv::Mat1s operator()() const { return labels; }

	/// Get label matrix, index 0 means "empty" label, indices are subsequent.
	const cv::Mat1s getLabels() const { return labels; }

	/// return number of label indices (including 0 -- background label)
	int getCount() const { return labelcount; }

	/** Set labels matrix.
		Use this function if your labels are already using subsequent indices
		starting with 0 (the "empty" label) or 1 (all pixels labeled). **/
	void setLabels(const cv::Mat &labeling);

	/** Read-in matrix containing arbitrary labels.
		No assumptions are made about the indices used, so the matrix is
	    examined in a rather expensive way to gain nice, subsequent indices.
		Note that label colors are always overwritten if the image is colored
		or their are not enough colors present.
		@arg binary Interpret as binary labeling, i.e. all values > 0 are
			 interpreted as label 1.
	**/
	void read(const cv::Mat &labeling, bool binary);

	cv::Mat1b grayscale() const;
	cv::Mat3b bgr() const;

	/** Obtain shiny label colors.
		@arg count The number of colors needed
		@arg yellowcursor When set, yellow is not used as label color.
	**/
	static std::vector<cv::Vec3b> colors(int count, bool yellowcursor);

	/** Obtain label colors of this labeling.
		If this labeling is based on a colored labeling image, these are custom.
	**/
	const std::vector<cv::Vec3b>& colors() const;

	/// Set colors used for label image generation
	void setColors(const std::vector<cv::Vec3b>& colors);

	/// Reserve yellow for interactive cursor color
	/** When set, yellow is not used as label color. Set by default. */
	bool yellowcursor;

protected:

	/// helper function to build labelColors based on labelcount
	void buildColors() const;

	mutable std::vector<cv::Vec3b> labelColors;
	cv::Mat1s labels;
	int labelcount;
};

}

#endif // WITH_OPENCV2
#endif // LABELING_H
