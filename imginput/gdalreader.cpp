#ifdef WITH_GDAL

#include "imginput.h"
#include "gdalreader.h"

#include <gdal_priv.h>
#include <cpl_conv.h>

#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>        // std::stable_sort

using std::vector;

namespace imginput {

// structure and compare-function to sort band metadata by wavelength
// maybe just copy the center value out, shorter than BandDesc struct...
struct indexMetaTuple
{
	int bandId; // GDAL-BandId (range [1, n])
	multi_img_base::BandDesc bandDesc;
};

// compares 2 indexMetaTuples by wavelength
// sort empty bandDescs in front of non-empty ones
bool tupleCompare(indexMetaTuple a, indexMetaTuple b)
{
	if (b.bandDesc.empty) return false; // nothing can be strictly less than empty
	if (a.bandDesc.empty) return true;  // empty (a) before non-empty (b)
	return a.bandDesc.center < b.bandDesc.center; // both non-empty -> order by wavelength
}

multi_img::ptr GdalReader::readFile()
{
	GDALDataset *dataset;

	static bool gdalRegistered = false;
	if (!gdalRegistered)
	{
		GDALAllRegister();
		gdalRegistered = true;
	}

	// find appropriate GDALDataType, necessary for reading band data
	GDALDataType gdalDataType;
	if (typeid(multi_img::Value) == typeid(double))
	{
		gdalDataType = GDT_Float64;
	}
	else if (typeid(multi_img::Value) == typeid(float))
	{
		gdalDataType = GDT_Float32;
	}
	else
	{
		return multi_img::ptr(new multi_img());
	}

	// Load Image
	CPLSetErrorHandler(CPLQuietErrorHandler); // supress error reporting
	dataset = (GDALDataset *)GDALOpen(config.file.c_str(), GA_ReadOnly);
	CPLSetErrorHandler(NULL); // remove error report supression

	if (dataset == NULL)
	{
		// print error message, if we just selected the wrong file (.hdr instead of .raw)
		int cplerrno = CPLGetLastErrorNo();
		const char* cplerrmsg = CPLGetLastErrorMsg();
		if (cplerrno == 1)
			std::cerr << "GDAL-ERROR " << cplerrno << ": " << cplerrmsg << std::endl;
		CPLErrorReset();

		return multi_img::ptr(new multi_img());
	}

	// Read Dataset Metadata
	// Starting with GDAL 1.10, use dataset->GetMetadata("ENVI"); for ENVI metadata...
	// http://www.gdal.org/frmt_various.html -> ENVI
	std::cout << "Reading image: " << dataset->GetDescription() << std::endl;
	//char **metadataList = dataset->GetMetadata();
	//if (metadataList == NULL)
	//{
	//	std::cout << "Dataset does not provide metadata." << std::endl;
	//}
	//else
	//{
	//	std::cout << "Start of Dataset metadata" << std::endl;
	//	for (int i = 0; metadataList[i] != NULL; i++)
	//	{
	//		// form of metadataList entry: "key=value"
	//		char cpy[strlen(metadataList[i])];
	//		char *key, *value;
	//		strcpy(cpy, metadataList[i]);
	//		key   = strtok(cpy,  "=");
	//		value = strtok(NULL, "=");

	//		std::cout << key << " = " << value << std::endl;
	//	}
	//	std::cout << "End of Dataset metadata" << std::endl << std::endl;
	//}

	// Find the ordering of the bands
	std::vector<indexMetaTuple> metaTuples(dataset->GetRasterCount());
	// bands are indexed from 1 to n [inclusive, inclusive]
	for (int bandNr = 1; bandNr <= dataset->GetRasterCount(); ++bandNr)
	{
		GDALRasterBand *band;

		band = dataset->GetRasterBand(bandNr);

		// read metadata
		multi_img_base::BandDesc bandDesc;

		switch(band->GetColorInterpretation())
		{
		case GCI_RedBand:
			bandDesc = multi_img_base::BandDesc(620);
			break;
		case GCI_GreenBand:
			bandDesc = multi_img_base::BandDesc(540);
			break;
		case GCI_BlueBand:
			bandDesc = multi_img_base::BandDesc(460);
			break;
		case GCI_PaletteIndex:
			std::cerr << "Error loading image: GDAL palette indices are not supported." << std::endl;
			return multi_img::ptr(new multi_img());
		case GCI_Undefined:
			{ // (limit scope of bool converted and float center)
				bool converted = false;
				float center;

				// try to parse the band description, hope its value is a float that describes the wavelength
				const char *desc = band->GetDescription();
				converted = tryConvert(desc, center);

				// try to parse an entry called Band_X, hope its value is a float that describes the wavelength
				if (!converted)
				{
					std::stringstream stream;
					stream << "Band_" << bandNr;
					const char *strval = dataset->GetMetadataItem(stream.str().c_str());

					// Entry exists
					if (strval != NULL)
					{
						converted = tryConvert(strval, center);
					}
				}

				// update band descriptor
				if (converted)
					bandDesc = multi_img_base::BandDesc(center);
				else
					bandDesc = multi_img_base::BandDesc();

				break;
			}
		default:
			std::cerr << "The format of band " << bandNr-1 << " is not supported. (Zero based index)" << std::endl;
			break;
		}

		metaTuples[bandNr - 1].bandId = bandNr;
		metaTuples[bandNr - 1].bandDesc = bandDesc;
	}

	// bands are not ordered by wavelength, e.g. could be reversed or even unordered
	// stable_sort: do not change the order of bands with unknown wavelength
	std::stable_sort(metaTuples.begin(), metaTuples.end(), tupleCompare);

	// find ROI
	int xOff = 0;
	int yOff = 0;
	int sizeX = dataset->GetRasterXSize();
	int sizeY = dataset->GetRasterYSize();
	if (!config.roi.empty())
	{
		// Do not print an error message here, as it will be printed in ImgInput anyways
		std::vector<int> roiVals;
		if (ImgInput::parseROIString(config.roi, roiVals))
		{
			xOff = roiVals[0];
			yOff = roiVals[1];
			sizeX = roiVals[2];
			sizeY = roiVals[3];
		}
	}

	// crop spectrum
	int bandlow = 0;
	int bandhigh = dataset->GetRasterCount() - 1; // inclusive, just like config.bandhigh
	if ((config.bandlow > 0) ||
		(config.bandhigh > 0 && config.bandhigh < dataset->GetRasterCount() - 1))
	{
		// if bandhigh is not specified, do not limit
		bandhigh = (config.bandhigh == 0) ? (dataset->GetRasterCount() - 1) : config.bandhigh;

		// correct input?
		if (config.bandlow > bandhigh || bandhigh > dataset->GetRasterCount() - 1)
		{
			std::cerr << "Inconsistent bandlow, bandhigh values specified!" << std::endl;
			return multi_img::ptr(new multi_img());
		}
		bandlow = config.bandlow;
	}

	// create multi_img & fill it with data
	multi_img::ptr img_ptr(new multi_img(
			sizeY,
			sizeX,
			bandhigh - bandlow + 1)); // bandhigh is inclusive

	double maxVal = 0;
	for (int metaDataIdx = bandlow; metaDataIdx <= bandhigh; ++metaDataIdx)
	{
		std::string desc = metaTuples[metaDataIdx].bandDesc.str();
		std::cout << "Reading band " << metaDataIdx;
		if (!desc.empty())
			std::cout << ": " << metaTuples[metaDataIdx].bandDesc.str();
		std::cout << std::endl;

		GDALRasterBand *band;
		double minMax[2]; // [0] min, [1] max

		band = dataset->GetRasterBand(metaTuples[metaDataIdx].bandId);

		// Read Band Metadata
		//std::cout << "Band" << bandNr << "'s description: " << band->GetDescription() << std::endl;
		//char **metadataList = band->GetMetadata();
		//if (metadataList == NULL)
		//{
		//	std::cout << "Band" << bandNr << " does not provide metadata." << std::endl;
		//}
		//else
		//{
		//	std::cout << "Start of Band" << bandNr << "'s metadata" << std::endl;
		//	for (int i = 0; metadataList[i] != NULL; i++)
		//	{
		//		// form of metadataList entry: key=value
		//		char *cpy, *key, *value;
		//		strcpy(cpy, metadataList[i]);
		//		key   = strtok(cpy,  "=");
		//		value = strtok(NULL, "=");

		//		std::cout << key << " = " << value << std::endl;
		//	}
		//	std::cout << "End of Band" << bandNr << "'s metadata" << std::endl << std::endl;
		//}

		// find max of band
		int gotMax;
		minMax[1] = band->GetMaximum(&gotMax);
		if (!gotMax)
			GDALComputeRasterMinMax((GDALRasterBandH)band, TRUE, minMax);

		// update global max
		if (minMax[1] > maxVal)
			maxVal = minMax[1];

		// read band data
		void *scanline;
		scanline = CPLMalloc(sizeof(multi_img::Value) * sizeX * sizeY);
		band->RasterIO(GF_Read,
					   xOff, yOff, sizeX, sizeY,
					   scanline, sizeX, sizeY,
					   gdalDataType,
					   0, 0);

		// copy (meta-)data to multi_img (multi_img indices are 0 based, metaDataIdx starts with bandlow)
		int multiImgBandIdx = metaDataIdx - bandlow;

		// copy band data to multi_img (we want a zero based index)
		multi_img::Band mat(sizeY, sizeX, (multi_img::Value *)scanline);
		mat = cv::max(mat, 0.);
		img_ptr->setBand(multiImgBandIdx, mat);
		CPLFree(scanline);

		img_ptr->meta[multiImgBandIdx] = metaTuples[metaDataIdx].bandDesc;
	}

	/* if our image data has more than 8 bit (values > 255), then
	 * determine dynamic range of camera (we assume it is a power of two) */
	double powMax = 256;
	for (; powMax < maxVal; powMax *= 2) {
		// nothing
	}
	maxVal = powMax;

	GDALClose(dataset);

	// set min & max
	img_ptr->minval = 0;
	img_ptr->maxval = (multi_img::Value)maxVal;

	/* invalidate pixel cache as pixel length has changed
	   This step is _mandatory_ also to initialize cache containers */
	img_ptr->resetPixels();

	return img_ptr;
}


bool GdalReader::tryConvert(std::string const& str, float& value)
{
	int reads = sscanf(str.c_str(), "%f", &value);
	return (reads == 1);
}

} //namespace

#endif // WITH_GDAL
