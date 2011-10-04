#include "SOMDatabase.h"

SOMDatabase::SOMDatabase(void)
{
	m_pSkin = NULL;
	m_pNonSkin = NULL;
}

SOMDatabase::~SOMDatabase(void)
{
	//release database
	if (m_pSkin != NULL)
		delete[] m_pSkin;
	if (m_pNonSkin != NULL)
		delete[] m_pNonSkin;
}

bool SOMDatabase::load(const char *cpDirectory, unsigned int uiNumPixelsSkin, unsigned int uiNumPixelsNonSkin, unsigned int uiPixelPerFile)
{
	//check params
	if (cpDirectory == NULL)
		return false;

	if (m_pSkin != NULL) {
		delete[] m_pSkin;
		m_pSkin = NULL;
	}
	if (m_pNonSkin != NULL) {
		delete[] m_pNonSkin;
		m_pNonSkin = NULL;
	}

	//store params and (re)allocate pixel arrays
	m_uiSkinPixels = uiNumPixelsSkin;
	m_uiNonSkinPixels = uiNumPixelsNonSkin;
	m_pSkin = new Color[m_uiSkinPixels];
	if (m_uiNonSkinPixels != 0) {
		m_pNonSkin = new Color[m_uiNonSkinPixels];
	}

	//load skin pixels
	if (!loadSkin(cpDirectory, uiPixelPerFile))
		return false;

	//if requested load nonskin pixels
	if (m_uiNonSkinPixels != 0) {
		if (!loadNonSkin(cpDirectory, uiPixelPerFile))
			return false;
	}

	return true;
}

void SOMDatabase::getSkin(unsigned int uiIndex, Color &color)
{
	color.R = m_pSkin[uiIndex].R;
	color.G = m_pSkin[uiIndex].G;
	color.B = m_pSkin[uiIndex].B;
}

void SOMDatabase::getNonSkin(unsigned int uiIndex, Color &color)
{
	color.R = m_pNonSkin[uiIndex].R;
	color.G = m_pNonSkin[uiIndex].G;
	color.B = m_pNonSkin[uiIndex].B;
}

unsigned int SOMDatabase::numSkinPixels()
{
	return m_uiSkinPixels;
}

unsigned int SOMDatabase::numNonSkinPixels()
{
	return m_uiNonSkinPixels;
}

bool SOMDatabase::loadSkin(const char* cpDirectory, unsigned int uiPixelPerFile)
{
	char buffer[512];

	//open file streams
	std::string maskList(cpDirectory);
	maskList.append("/filenames_masks.txt");
	std::string skinList(cpDirectory);
	skinList.append("/filenames_skin.txt");

	std::fstream maskStream;
	std::fstream skinStream;

	maskStream.open(maskList.c_str(), std::ios_base::in);
	skinStream.open(skinList.c_str(), std::ios_base::in);
	if (!maskStream.is_open() || !skinStream.is_open()) {
		return false;
	}

	//go trough file and read skin and mask files
	unsigned int skinLoaded = 0;
	while (skinLoaded < m_uiSkinPixels) {
		if (maskStream.eof()  || skinStream.eof()) {
			m_uiSkinPixels = skinLoaded;
			break;
		}
		maskStream.getline(buffer, 512);
		//check if filname is plausible (shortest possible eg. x.jpg)
		if (strlen(buffer) < 5) continue;
		
		//put path and name together
		std::stringstream ssMask;
		ssMask << cpDirectory << "/masks/" << buffer;
		skinStream.getline(buffer, 512);
		std::stringstream ssSkin;
		ssSkin << cpDirectory << "/skin-images/" << buffer;

		//calculate number of pixels to load from current file
		unsigned int pixelToLoad = uiPixelPerFile;
		if (skinLoaded+uiPixelPerFile > m_uiSkinPixels) pixelToLoad = m_uiSkinPixels-skinLoaded;
		skinLoaded += loadSkinFile(ssMask.str().c_str(), ssSkin.str().c_str(), pixelToLoad, skinLoaded);
	}
	maskStream.close();
	skinStream.close();
	return true;
}

int SOMDatabase::loadSkinFile(const char* maskFile, const char* skinFile, unsigned int uiPixels, unsigned int uiOffset)
{
	//load image and mask		
	IplImage* mask = cvLoadImage(maskFile);
	if (mask->imageData == NULL) {
		return 0;
	}
	IplImage* img = cvLoadImage(skinFile);
	if (img->imageData == NULL) {
		return 0;
	}

	//go through image
	unsigned int pixelCounter = 0;
	for (int i = 0; i < mask->height; ++i) {
		for (int j = 0; j < mask->width; ++j) {
			//check if current pixel is masked => skin
			if (mask->imageData[i*mask->widthStep+j*3] != 0) {
				m_pSkin[uiOffset+pixelCounter].B = img->imageData[i*img->widthStep+j*3];
				m_pSkin[uiOffset+pixelCounter].G = img->imageData[i*img->widthStep+j*3+1];
				m_pSkin[uiOffset+pixelCounter].R = img->imageData[i*img->widthStep+j*3+2];

				//bookkeeping
				++pixelCounter;
				if (pixelCounter == uiPixels) {
					cvReleaseImage(&mask);
					cvReleaseImage(&img);
					return pixelCounter;
				}
			}
		}
	}
	cvReleaseImage(&mask);
	cvReleaseImage(&img);
	return pixelCounter;
}

bool SOMDatabase::loadNonSkin(const char *cpDirectory, unsigned int uiPixelPerFile)
{
	char buffer[512];

	//open file stream
	std::string nonSkinList(cpDirectory);
	nonSkinList.append("/filenames_nonskin.txt");

	std::fstream nonSkinStream;

	nonSkinStream.open(nonSkinList.c_str(), std::ios_base::in);
	if (!nonSkinStream.is_open()) {
		return false;
	}

	//read filenames from stream
	unsigned int nonSkinLoaded = 0;
	while (nonSkinLoaded < m_uiNonSkinPixels) {
		if (nonSkinStream.eof()) {
			m_uiNonSkinPixels = nonSkinLoaded;
			break;
		}
		nonSkinStream.getline(buffer, 512);
		//check if filname is plausible (shortest possible eg. x.jpg)
		if (strlen(buffer) < 5) continue;
		
		//put path and name together
		std::stringstream ssNonSkin;
		ssNonSkin << cpDirectory << "/non-skin-images/" << buffer;

		//calculate number of pixels to load
		unsigned int pixelToLoad = uiPixelPerFile;
		if (nonSkinLoaded+uiPixelPerFile > m_uiNonSkinPixels) pixelToLoad = m_uiNonSkinPixels-nonSkinLoaded;
		nonSkinLoaded += loadNonSkinFile(ssNonSkin.str().c_str(), pixelToLoad, nonSkinLoaded);
	}
	nonSkinStream.close();
	return true;
}

int SOMDatabase::loadNonSkinFile(const char *nonSkinFile, unsigned int uiPixels, unsigned int uiOffset)
{
	//load image
	IplImage* img = cvLoadImage(nonSkinFile);
	if (img->imageData == NULL) {
		return 0;
	}
	//go trough image
	unsigned int pixelCounter = 0;
	for (int i = 0; i < img->height; ++i) {
		for (int j = 0; j < img->width; ++j) {
			m_pNonSkin[uiOffset+pixelCounter].B = img->imageData[i*img->widthStep+j*3];
			m_pNonSkin[uiOffset+pixelCounter].G = img->imageData[i*img->widthStep+j*3+1];
			m_pNonSkin[uiOffset+pixelCounter].R = img->imageData[i*img->widthStep+j*3+2];
			//bookkeeping
			++pixelCounter;
			if (pixelCounter == uiPixels) {
				cvReleaseImage(&img);
				return pixelCounter;
			}
		}
	}
	cvReleaseImage(&img);
	return pixelCounter;
}
