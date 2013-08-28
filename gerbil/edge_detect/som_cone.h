#ifndef SOMCONE_H
#define SOMCONE_H

#include "som.h"

class SOMCone : public SOM
{
protected:
	// Iterates over all neurons
	class IteratorCone : public SOM::IteratorBase
	{
	public:
		IteratorCone(SOMCone *base, int idx) : SOM::IteratorBase(),
			base(base), idx(idx) { }
		~IteratorCone() { }

		void operator++();
		Neuron &operator*() { return base->neurons[idx]; }
		IteratorCone *clone() { return new IteratorCone(*this); }

		// returns the coordinates of the neurons in the multi_img
		// that is created by export_2d
		cv::Point get2dCoordinates() { return cv::Point(idx, 0); }

		cv::Point3d get3dPos() { return base->coordinates[idx]; }

		SOM::neighbourIterator neighboursBegin(double radius);
		SOM::neighbourIterator neighboursEnd(double radius);

		// sollte eig nur in SOMCone per friend-klasse zugreifbar sein...
		inline int getIdx() const { return idx; }

	protected:
		bool equal(const SOM::IteratorBase &other) const;

	private:
		SOMCone *const base;
		int idx;
	};

	// Iterates over the neighbours of a neuron in a given radius
	class NeighbourIteratorCone : public SOM::NeighbourIteratorBase
	{
	public:
		NeighbourIteratorCone(SOMCone *base, int neuronIdx,
							double radius, bool end = false);
		~NeighbourIteratorCone() { }

		void operator++();
		Neuron &operator*() { return base->neurons[idx]; }
		NeighbourIteratorCone *clone() { return new NeighbourIteratorCone(*this); }

		double getDistance() const;
		double getDistanceSquared() const;
		double getFakeGaussianWeight(double sigma) const;

	protected:
		bool equal(const SOM::NeighbourIteratorBase &other) const;

	private:
		SOMCone *const base;
		int idx, nextSlice;
		const cv::Point3d &neuron;
		const double radius;
	};

	class CacheCone : public Cache
	{
	public:
		/**
		  @arg som Already trained Self-Organizing map
		  @arg img_height height of the pixel cache (image size, not som size)
		  @arg img_width width of the pixel cache (image size, not som size)
		  */
		CacheCone(SOMCone *som, int img_height, int img_width) : Cache(), som(som),
			data(std::vector<std::vector<int> >(img_height,
				std::vector<int>(img_width, -1))) { }

		void preload(const multi_img &img);

		double getDistance(const multi_img::Pixel &v1,
						   const multi_img::Pixel &v2,
						   const cv::Point &c1,
						   const cv::Point &c2);

		double getSobelX(int x, int y);
		double getSobelY(int x, int y);

	private:
		// shortcut to remove clutter in code
		inline cv::Point3d coords(int y, int x) {
			return som->coordinates[data[y][x]];
		}

		SOMCone *const som;
		std::vector<std::vector<int> > data; // pointer in som->coordinates array
	};

public:
	SOMCone(const vole::EdgeDetectionConfig &conf, int dimension);
	SOMCone(const vole::EdgeDetectionConfig &conf, const multi_img &data);

	void initCoordinates(int dimension);

	/**
	  @arg img_height height of the pixel cache (image size, not som size)
	  @arg img_width width of the pixel cache
	  */
	CacheCone *createCache(int img_height, int img_width);

	int size() const { return neurons.size(); }

	int get2dWidth() const { return neurons.size(); }
	int get2dHeight() const { return 1; }

	SOM::iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);

	/*int updateNeighborhood(iterator &neuron,
							const multi_img::Pixel &input,
							double sigma, double learnRate);*/

	double getDistanceBetweenWinners(const multi_img::Pixel &v1,
									 const multi_img::Pixel &v2);

	std::string toString();

	iterator begin();
	iterator end();

protected:
	double getDistance(int idx1, int idx2);
	static double getDistance(cv::Point3d &p1, cv::Point3d &p2);
	double getDistanceSquared(int idx1, int idx2);
	static double getDistanceSquared(cv::Point3d &p1, cv::Point3d &p2);

	std::vector<Neuron> neurons;			///< Neurons in the SOM
	// we don't neccessarily need to store the z coordinate for each neuron on a
	// slice individually, but we don't have direct acess to the slice number
	std::vector<cv::Point3d> coordinates;	///< 3d Coordinates of the Neurons
	double granularity;

private:
	int slices;					///< Slices of SOM cone
	std::vector<int> slicePtr;	///< Index of first Neuron of a slice (slicePtr[slices] == #neurons)
	// the z coordinates of the slices have to be ordered (currently asc or desc)
	std::vector<double> sliceZ;	///< z coordinate of the slice
};

#endif // SOMCONE_H
