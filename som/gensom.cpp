#include "gensom.h"
#include "isosom.h"

#include <progress_observer.h>

#include <similarity_measure.h>
#include <sm_factory.h>

#include <opencv2/highgui/highgui.hpp> // for debug writeout
#include <boost/cstdint.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <algorithm>
#include <functional>

namespace som {

/** Training **/

void GenSOM::train(const multi_img &input, ProgressObserver *po)
{
	std::cout << "Start feeding" << std::endl;

	// matrices hold shuffled sequences of the input for number of iterations
	int maxIter = config.maxIter;
	cv::Mat_<int> shuffledY(1, maxIter);
	cv::Mat_<int> shuffledX(1, maxIter);

	// generate random sequence of the input x,y range
	cv::RNG rng(config.seed);
	rng.fill(shuffledY, cv::RNG::UNIFORM,
			 cv::Scalar(0), cv::Scalar(input.height));
	rng.fill(shuffledX, cv::RNG::UNIFORM,
			 cv::Scalar(0), cv::Scalar(input.width));

	// output percentage
	unsigned int hundred = std::max<unsigned int>(maxIter/100, 100);
	int percent = 1;
	if (config.verbosity > 0) {
		std::cout << "  0 %";
		std::cout.flush();
	}
	long sumOfUpdates = 0;

	// starting training (notifier needed by OpenCL impl.)
	notifyTrainingStart();

	cv::MatConstIterator_<int> itY = shuffledY.begin();
	cv::MatConstIterator_<int> itX = shuffledX.begin();
	for (int curIter = 0; curIter < maxIter; ++curIter, ++itX, ++itY)
	{
		// feed one sample
		const multi_img::Pixel &vec = input(*itY, *itX);
		sumOfUpdates += trainSingle(vec, curIter, maxIter);

		// print progress (and maybe exit)
		if ((config.verbosity > 0 || po) && (config.maxIter > 100)
			&& ((curIter % hundred) == 0) && (percent < 100)) {

			// send progress updates to observer or stdout
			if (po) {
				bool cont = po->update(curIter / (float)config.maxIter);
				if (!cont) {
					std::cerr << "Aborting training" << std::endl;
					return;
				}
			} else {
				std::cout << "\r " << (percent < 10 ? " " : "")
						  << percent << " %";
				std::cout.flush();
			}
			if (config.verbosity >= 2) {
				std::cout << " Feed #" << curIter
						  << ", avg. updates per iteration in the last "
						  << (maxIter / hundred) << " iterations: "
						  << ((double)sumOfUpdates / hundred)
						  << std::endl;
				sumOfUpdates = 0;
			}

			// print som each 20%
			if (config.verbosity >= 3 && (percent % 20) == 0)
			{
				// TODO
				/*multi_img somimg = som->export_2d();
				std::ostringstream filename, filename2;
				filename << "debug_som_";
				filename << std::setfill('0') << std::setw(2);
				filename << (ctr / (config.maxIter / 20));
				somimg.write_out(filename.str());
				filename2 << filename.str();

				filename << "-rgb.png";
				cv::imwrite(filename.str(), somimg.bgr() * 255.f);

				if (!updateMap.empty()) {
					cv::Mat3f upBGR(updateMap.rows, updateMap.cols);
					cv::cvtColor(updateMap, upBGR, CV_GRAY2BGR);
					for (cv::Mat3f::iterator it = upBGR.begin(); it != upBGR.end();++it)
						if (*it == cv::Vec3f(0.f))
							*it = cv::Vec3f(0.f, 0.f, 1.f);

					filename2 << "-updates.png";
					cv::imwrite(filename2.str(), upBGR * 255.f);
				}*/
			}

			percent++;
		}
	}
	if (config.verbosity > 0)
		std::cout << "\r100 %" <<std::endl;

	// finished training (notifier needed by OpenCL impl.)
	notifyTrainingEnd();

	std::cout <<"# Feeding done" <<std::endl;

	if (!config.somFile.empty()) {
		std::cout << "# writing SOM in binary format to \""
				  << config.somFile << "\"" << std::endl;
		saveFile(config.somFile);
	}
    if (config.verbosity > 2) {
        cv::imwrite("debug_som.png", bgr(input.meta, input.maxval)*255.f);
    }
 }

int GenSOM::trainSingle(const multi_img::Pixel &input, int iter, int max)
{
	// adjust learning rate and radius
	// note that they are _decreasing_ -> start * (end/start)^(iter%)
	double learnRate = config.learnStart * std::pow(
				config.learnEnd / config.learnStart,
				(double)iter/(double)max);
	double sigma = config.sigmaStart * std::pow(
				config.sigmaEnd / config.sigmaStart,
				(double)iter/(double)max);

	// find best matching unit to given input vector
	size_t index = findBMU(input).index;

	// increase winning count of neuron
	//m_bmuMap(pos) += 1.0;

	int updates = updateNeighborhood(index, input, sigma, learnRate);

	return updates;
}

double GenSOM::gaussWeight(double distance, double sigma, double learnRate)
{
	double gaussian = exp(-(distance) / (2.0*sigma*sigma));
	return learnRate * gaussian;
}


/** I/O **/

#ifdef BOOST_LITTLE_ENDIAN
template <typename T>
T swap_endian(const T& u) {
	return u;
}
#else /* BOOST_LITTLE_ENDIAN */
template <typename T>
T swap_endian(const T& u)
{
	union
	{
		T u;
		unsigned char u8[sizeof(T)];
	} source, dest;

	source.u = u;

	for (size_t k = 0; k < sizeof(T); k++)
		dest.u8[k] = source.u8[sizeof(T) - k - 1];

	return dest.u;
}
#endif /* BOOST_LITTLE_ENDIAN */

// write data in little endian order
template <typename T>
void writeLittle(std::ostream &os, const T& x)
{
	const T le = swap_endian(T(x));
	os.write(reinterpret_cast<const char *>(&le), sizeof(T));
}

GenSOM::GenSOM(const SOMConfig &config)
	: config(config),
	  distfun(similarity_measures::SMFactory<value_type>::
			  spawn(config.similarity))
{}

void GenSOM::init(size_t nneurons, size_t nbands, bool randomize)
{
	// create vector contents
	neurons.assign(nneurons, Neuron(nbands));

	if (randomize) {
		// initialize randomly. Note that initialization range does not matter.
		cv::RNG rng(config.seed);

		for (size_t i = 0; i < neurons.size(); ++i) {
			neurons[i].randomize(rng, 0., 1.);
		}
	}
}

GenSOM *GenSOM::create(const SOMConfig &conf, size_t nbands, bool randomize)
{
	if (SOM_SQUARE == conf.type) {
		return new IsoSOM<2>(conf, nbands, randomize);
	} else if (SOM_CUBE == conf.type) {
		return new IsoSOM<3>(conf, nbands, randomize);
	} else if (SOM_TESSERACT == conf.type) {
		return new IsoSOM<4>(conf, nbands, randomize);
	} else {
		std::stringstream ss;
		ss << "GenSOM::create(): " << conf.type << " not implemented."
		   << std::endl;
		throw std::runtime_error(ss.str());
	}
}

GenSOM::~GenSOM()
{
	delete distfun;
}

GenSOM *GenSOM::create(const SOMConfig &conf, const multi_img &img,
					   ProgressObserver *po)
{
	GenSOM *som;

	/* If the SOM is already trained and on disk, load it */
	const bool fileExists = boost::filesystem::exists(conf.somFile);
	if (!conf.somFile.empty() && fileExists) {
		// An existing file is assumed to be a valid SOM file. Otherwise
		// this is an error and loadFile() will throw.
		std::cout << "# loading SOM from binary file "
				  << "\"" << conf.somFile << "\" ... ";
		som = loadFile(conf.somFile, conf);
		std::cout << "done" << std::endl;
	} else {
		/* we need to train. Let the user know if he specified a filename */
		if (!conf.somFile.empty() && !fileExists) {
			std::cout << "# warning: SOM binary file "
					  << "\"" << conf.somFile << "\" "
					  << "does not exist. Starting training."
					  << std::endl;
		}
		som = create(conf, img.size(), /* randomize */ true);
		som->train(img, po);
	}
	return som;
}

DistIndexPair
GenSOM::findBMU(const multi_img::Pixel &inputVec) const
{
	// the best matching unit (index and distance to input) we want to find
	DistIndexPair bmu;

	for (size_t idx = 0; idx < neurons.size(); ++idx) {
		const double dist = distfun->getSimilarity(neurons[idx], inputVec);
		if (dist < bmu.dist) {
			bmu.dist = dist;
			bmu.index = idx;
		}
	}
	return bmu;
}

void GenSOM::saveFile(std::ostream &os) const
{
	if (!os) {
		throw std::runtime_error("GenSOM::saveFile(): "
								 "could not write to stream.");
	}

	os << "gerbilsom\x20\x20\x20\x20\x20\x20\x20";  // 16 byte "magic"
	writeLittle<int32_t>(os, 3);                    // file version
	writeLittle<int32_t>(os, 1);                    // data type: 1 = ieee float
	writeLittle<int32_t>(os, int32_t(config.type));   // SOM type
	writeLittle<int32_t>(os, int32_t(neurons.size()));   // SOM size
	int nbands;
	if (neurons.size() > 0) {
		nbands = neurons[0].size();
	} else {
		nbands = 0;
	}
	writeLittle<int32_t>(os, int32_t(nbands));   // num bands


	if (!os) {
		throw std::runtime_error("GenSOM::saveFile(): "
								 "could not write to stream.");
	}

	// write out neurons
	for (std::vector<Neuron>::const_iterator vit = neurons.begin();
		 vit != neurons.end();
		 ++vit)
	{
		for (Neuron::const_iterator nit = vit->begin();
			 nit != vit->end();
			 ++nit)
		{
			writeLittle(os, *nit);
		}
	}

	if (!os) {
		throw std::runtime_error("GenSOM::saveFile(): "
								 "could not write to stream.");
	}
}

void GenSOM::saveFile(const std::string &fileName) const
{
	try {
		std::ofstream os(fileName.c_str(), std::ios::out | std::ios::binary);
		saveFile(os);
	} catch (const std::exception& e) {
		std::stringstream ss;
		ss << "GenSOM::saveFile(): While trying to save to file '"
		   << fileName << "': " << e.what();
		throw std::runtime_error(ss.str());
	}
}

// read data stored in little endian
template <typename T>
T readLittle(std::istream &is)
{
	T x;
	is.read(reinterpret_cast<char *>(&x), sizeof(T));
	x = swap_endian(x);
	return x;
}

GenSOM *GenSOM::loadFile(std::istream &is, const SOMConfig& config)
{

	const std::string magic("gerbilsom\x20\x20\x20\x20\x20\x20\x20");
	char readmagic[16+1];
	is.read(readmagic, 16);
	readmagic[16] = '\0';
	if(!is) {
		std::stringstream ss;
		ss << "GenSom::loadFile(): could not read from stream";
		throw std::runtime_error(ss.str());
	}

	if(magic != std::string(readmagic)) {
		throw std::runtime_error("GenSom::loadFile(): bad som file");
	}
	int32_t version = readLittle<int32_t>(is);
	if(version != 3) {
		throw std::runtime_error("GenSom::loadFile(): bad version som file");
	}

	int32_t datatype = readLittle<int32_t>(is);
	if(datatype != 1) {
		std::stringstream ss;
		ss << "GenSom::loadFile(): bad datatype, expected 1 (float), got "
		   << datatype;
		throw std::runtime_error(ss.str());
	}

	int32_t type = readLittle<int32_t>(is);
	if (config.type != type) {
		std::stringstream ss;
		ss << "GenSom::loadFile(): "
		   << "stored SOM number of dimensions "
		   << type <<  " does not match config type="
		   << config.type;
		throw std::runtime_error(ss.str());
	}
	int32_t size = readLittle<int32_t>(is);
	// Note: need to create SOM before we can perform size sanity check
	int32_t nbands = readLittle<int32_t>(is);

	GenSOM* som = create(config, nbands, /* randomize */ false);
	size_t nneurons = som->neurons.size();

	if (nneurons != (size_t)size) {
		std::stringstream ss;
		ss << "GenSom::loadFile(): "
		   << "stored SOM dimension size "
		   << size <<  " does not match size of this config's SOM = "
		   << nneurons;
		throw std::runtime_error(ss.str());
	}

	for (size_t i=0; i<nneurons; ++i) {
		Neuron &ne = som->neurons[i];
		for (int j=0; j<nbands; ++j) {
			ne[j] = readLittle<float>(is);
		}
	}
	return som;
}

GenSOM *GenSOM::loadFile(const std::string &fileName, const SOMConfig& config)
{
	GenSOM *som = 0;
	try {
		std::ifstream is(fileName.c_str(), std::ios::in | std::ios::binary);
		som = loadFile(is, config);
	} catch (const std::exception& e) {
		if(som)
			delete som;
		std::stringstream ss;
		ss << "GenSOM::loadFile(): While trying to load file '"
		   << fileName << "': " << e.what();
		throw std::runtime_error(ss.str());
	}
	return som;
}

multi_img GenSOM::img(const std::vector<multi_img_base::BandDesc> &meta,
					  const multi_img_base::Range &range)
{
	cv::Size size = size2D();
	multi_img ret(size.height, size.width, neurons[0].size());
	ret.meta = meta;
	ret.minval = range.min; ret.maxval = range.max;
	for (size_t i = 0; i < neurons.size(); ++i) {
		ret.setPixel(getCoord2D(i), neurons[i]);
	}
	return ret;
}

cv::Mat3f GenSOM::bgr(const std::vector<multi_img_base::BandDesc> &meta,
					  multi_img_base::Value maxval)
{
	cv::Mat3f ret(size2D());
	for (size_t i = 0; i < neurons.size(); ++i) {
		ret(getCoord2D(i)) = multi_img::bgr(neurons[i], meta, maxval);
	}
	return ret;
}

}
