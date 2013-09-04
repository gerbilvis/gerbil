#ifndef SOM3D_H
#define SOM3D_H

#include "som.h"
#include <tbb/parallel_for.h>

class SOM3d : public SOM
{
public:
	// Iterates over all neurons
	class Iterator3d : public SOM::IteratorBase
	{
	public:
		Iterator3d(SOM3d *base, int x, int y, int z) : SOM::IteratorBase(),
			base(base), x(x), y(y), z(z) { }
		~Iterator3d() { }

		void operator++();
		Neuron &operator*() { return base->neurons[z][y][x]; }
		Iterator3d *clone() { return new Iterator3d(*this); }

                void assignWith(const IteratorBase *source) {
                        // types have been checked by caller
                        const Iterator3d *s = static_cast<const Iterator3d *>(source);
                        *this = *s;
                }

		// returns the coordinates of the neurons in the multi_img
		// that is created by export_2d
		cv::Point get2dCoordinates() { return cv::Point(x + y * base->width, z); }

		cv::Point3d get3dPos() { return cv::Point3d(x, y, z); }

		SOM::neighbourIterator neighboursBegin(double radius);
		SOM::neighbourIterator neighboursEnd(double radius);

		// sollten eig nur in SOM3d per friend-klasse zugreifbar sein...
		inline cv::Point3i getId() const { return cv::Point3i(x, y, z); }

	protected:
		bool equal(const SOM::IteratorBase &other) const;

	private:
		SOM3d *base;
		int x, y, z;
	};

	// Iterates over the neighbours of a neuron in a given radius
	class NeighbourIterator3d : public SOM::NeighbourIteratorBase
	{
	public:
		NeighbourIterator3d(SOM3d *base, cv::Point3i neuron,
							double radius, bool end = false);
		~NeighbourIterator3d() {}

		void operator++();
		Neuron &operator*() { return base->neurons[z][y][x]; }
		NeighbourIterator3d *clone() { return new NeighbourIterator3d(*this); }

		double getDistance() const;
		double getDistanceSquared() const;
		double getFakeGaussianWeight(double sigma) const;

	protected:
		bool equal(const SOM::NeighbourIteratorBase &other) const;

	private:
		SOM3d *const base;
		const int fromX, toX, fromY, toY, toZ;
		int x, y, z;
		const cv::Point3i neuron;
		const double radiusSquared;
	};

	class Cache3d : public Cache
	{
	public:
		struct PreloadTBB
		{
			PreloadTBB(const multi_img &image, SOM3d *som,
					   std::vector<std::vector<cv::Point3i> > &data)
					   : image(image), som(som), data(data) { }

			void operator()(const tbb::blocked_range<int>& r) const;

			const multi_img &image;
			SOM3d *const som;
			std::vector<std::vector<cv::Point3i> > &data;
		};


		/**
		  @arg som Already trained Self-Organizing map
		  @arg img_height height of the pixel cache (image size, not som size)
		  @arg img_width width of the pixel cache
		  */
		Cache3d(SOM3d *som, int img_height, int img_width) : Cache(), som(som),
			data(std::vector<std::vector<cv::Point3i> >(img_height,
				std::vector<cv::Point3i>(img_width, cv::Point3i(-1, -1, -1))))
		{ }

		void preload(const multi_img &img);

		double getDistance(const multi_img::Pixel &v1,
						   const multi_img::Pixel &v2,
						   const cv::Point &c1,
						   const cv::Point &c2);

		SOM::iterator getWinnerNeuron(cv::Point p);
		double getSimilarity(vole::SimilarityMeasure<multi_img::Value> *distfun,
		                     const cv::Point &p1,
		                     const cv::Point &p2);
		double getSobelX(int x, int y);
		double getSobelY(int x, int y);

	private:
		SOM3d *const som;
		std::vector<std::vector<cv::Point3i> > data;
	};

public:
	typedef std::vector<Neuron> Row;
	typedef std::vector<Row> Field;
	typedef std::vector<Field> Cube;
	friend class Cache3d;

	SOM3d(const vole::EdgeDetectionConfig &conf, int dimension,
		  std::vector<multi_img_base::BandDesc> meta);
	SOM3d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
		  std::vector<multi_img_base::BandDesc> meta);

	/**
	  @arg img_height height of the pixel cache (image size, not som size)
	  @arg img_width width of the pixel cache
	  */
	Cache3d *createCache(int img_height, int img_width);

	int size() const { return width * height * depth; }

	int get2dWidth() const { return width * height; }
	int get2dHeight() const { return depth; }

	iterator identifyWinnerNeuron(const multi_img::Pixel &inputVec);

	int updateNeighborhood(iterator &neuron,
							const multi_img::Pixel &input,
							double sigma, double learnRate);

	double getDistanceBetweenWinners(const multi_img::Pixel &v1,
									 const multi_img::Pixel &v2);

	cv::Vec3f getColor(cv::Point3d pos);

	std::string description();

	iterator begin();

protected:
	static double getDistance(const cv::Point3i &p1, const cv::Point3i &p2);
	static double getDistance(const cv::Point3d &p1, const cv::Point3d &p2);
	static double getDistanceSquared(const cv::Point3i &p1, const cv::Point3i &p2);
	static double getDistanceSquared(const cv::Point3d &p1, const cv::Point3d &p2);

private:
	template<bool mirrorZ>
	int updateSquare(const multi_img::Pixel &input,
					  const cv::Point3i &pos,
					  int deltaZ,
					  double sigmaSquare,
					  double learnRate);

protected:
	Cube neurons;				///< Neurons in the SOM grid

private:
	const int width;			///< Width  of SOM cube
	const int height;			///< Height of SOM cube
	const int depth;			///< Depth  of SOM cube
};

#endif // SOM3D_H
