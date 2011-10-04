/**
 * @file SOMDatabase.h
 * @brief Database for training of the Self-Organizing Maps,
 * @author Christoph Malskies
 * @date January 2010 
 */


#ifndef SOMDATABASE_H
#define SOMDATABASE_H

#include "defines.h"
#include <fstream>
#include <iostream>

#include "cv.h"
#include "highgui.h"

class SOMDatabase
{
public:
	SOMDatabase(void);
	~SOMDatabase(void);

	/** Loads the given number of skin and nonskin pixels from given databse.
	    @param cpDirectory Path to the database.
	    @param uiNumPixelsSkin Number of skin pixels to load.
	    @param uiNumPixelsNonSkin Number of nonskin pixels to load.
	    @param uiPixelPerFile Number of pixels loaded per file (for wider color spectrum).
	    @return True on success, false on error.
	*/
	bool load(const char* cpDirectory, unsigned int uiNumPixelsSkin, unsigned int uiNumPixelsNonSkin, unsigned int uiPixelPerFile);
	
	/** Retrieves a skin color value.
	    @param uiIndex Index of color value (0 - numSkinPixels()).
	    @param color Color value is stored here.
	*/
	void getSkin(unsigned int uiIndex, Color &color);
	/** Retrieves a nonskin color value.
	    @param uiIndex Index of color value (0 - numNonSkinPixels()).
	    @param color Color value is stored here.
	*/
	void getNonSkin(unsigned int uiIndex, Color &color);

	/** Returns the number of skin color pixels in database.
	    @return Number of skin pixels.
	*/
	unsigned int numSkinPixels();
	/** Returns the number of nonskin color pixels in databse.
	    @return Number of nonskin pixels.	
	*/
	unsigned int numNonSkinPixels();

private:
	Color* m_pSkin;
	Color* m_pNonSkin;
	
	unsigned int m_uiSkinPixels;
	unsigned int m_uiNonSkinPixels;

	//loads the filenames from list
	bool loadSkin(const char* cpDirectory, unsigned int uiPixelPerFile);
	//loads the pixels from one file
	int loadSkinFile(const char* maskFile, const char* skinFile, unsigned int uiPixels, unsigned int uiOffset);

	//loads the filenames from list
	bool loadNonSkin(const char* cpDirectory, unsigned int uiPixelPerFile);
	//loads the pixels from one file
	int loadNonSkinFile(const char* nonSkinFile, unsigned int uiPixels, unsigned int uiOffset);
};

#endif //SOMDATABASE_H
