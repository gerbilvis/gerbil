
template<typename T>
void PixelOperations::getChromaticitiesVector(const std::vector<cv::Vec<T, 3> > &rgbVec, std::vector<cv::Vec3d > &chromVec) {
	unsigned int nPixels = rgbVec.size();
	chromVec.assign(rgbVec.size(), cv::Vec3d());
	for (unsigned int i = 0; i < nPixels; ++i) {
		double sum = rgbVec[i][0] + rgbVec[i][1] + rgbVec[i][2];
		for (int j = 0; j < 3; ++j)
			chromVec[i][j] = static_cast<double>(rgbVec[i][j]) / sum;
	}
}


template<typename T, typename U>
void PixelOperations::getVectorPixelSum(const std::vector<cv::Vec<T, 3> > &rgbVec, std::vector<U> &sumVec) {
	sumVec.assign(rgbVec.size(), 0);
	for (unsigned int i = 0; i < rgbVec.size(); ++i) {
		sumVec[i] = rgbVec[i][0] + rgbVec[i][1] + rgbVec[i][2];
	}
}


template<typename T>
void PixelOperations::getMinMax(std::vector<T> &pixels, T &min, T &max) {
	min = DataTraits<T>::maxValue();
	max = DataTraits<T>::minValue();
	for (
		typename std::vector<T>::const_iterator it = pixels.begin();
		it != pixels.end();
		it++)
	{
		if (*it < min) min = *it;
		if (*it > max) max = *it;
	}
}




template<typename T>
void PixelOperations::meanPerChannel(const std::vector<cv::Vec<T, 3> > &inputVec, double &meanRed, double &meanGreen, double &meanBlue) {
	meanRed = meanGreen = meanBlue = 0;
	unsigned int count = 0;
	for (unsigned int i = 0; i < inputVec.size(); i++) {
		meanRed   += inputVec[i][0];
		meanGreen += inputVec[i][1];
		meanBlue  += inputVec[i][2];
		count++;
	}
	if (count == 0) count = 1;
	meanRed /= count;
	meanGreen /= count;
	meanBlue /= count;
}


template<typename T>
void PixelOperations::stdDevPerChannel(const std::vector<cv::Vec<T, 3> > &inputVec, double &stdDevRed, double &stdDevGreen, double &stdDevBlue) {

	double meanRed, meanGreen, meanBlue, tmp;
	stdDevRed = stdDevGreen = stdDevBlue = 0;
	meanPerChannel<T>(inputVec, meanRed, meanGreen, meanBlue);

	unsigned int nPixels = inputVec.size();
    if (nPixels == 0) return;
	for (unsigned int i = 0; i < nPixels; i++) {
		tmp = inputVec[i][0] - meanRed;   stdDevRed   += tmp*tmp;
		tmp = inputVec[i][1] - meanGreen; stdDevGreen += tmp*tmp;
		tmp = inputVec[i][2] - meanBlue;  stdDevBlue  += tmp*tmp;
	}
	stdDevRed   = std::sqrt(stdDevRed /= nPixels);
	stdDevGreen = std::sqrt(stdDevGreen /= nPixels);
	stdDevBlue  = std::sqrt(stdDevBlue /= nPixels);
}


template<typename V>
class pixel_operation_mycomparison
{
public:
	bool operator() (const cv::Vec<V, 3> &lhs, const cv::Vec<V, 3> &rhs) const
	{
		for (int i = 0; i < 3; ++i)
			if (lhs[i] != rhs[i]) return rhs[i]>lhs[i];
		return false;
	}
};


template<typename T>
void PixelOperations::removeDuplicates(
	std::vector<cv::Vec<T, 3> > &pixel_list,
	std::vector<cv::Vec<T, 3> > &unique_pixels)
{
	unsigned int nPixels = pixel_list.size();
	int count = 0;
	unique_pixels.assign(nPixels, cv::Vec<T, 3>());
	// if necessary, remove duplicated pixels
	// first, sort the vector:
	std::sort(pixel_list.begin(), pixel_list.end(), pixel_operation_mycomparison<T>());
	cv::Vec<T, 3> lastSeen(0, 0, 0);
	for (unsigned int i = 0; i < nPixels; ++i) {
		if (pixel_list[i] != lastSeen) {
			cv::Vec<T, 3> lastSeen = pixel_list[i];
			unique_pixels[count] = lastSeen;
			count++;
		}
	}
}
