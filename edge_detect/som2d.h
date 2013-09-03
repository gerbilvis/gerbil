#ifndef SOM2D_H
#define SOM2D_H

#include "som.h"
#include <tbb/parallel_for.h>

class SOM2d : public SOM
{
protected:
	// Iterates over all neurons
	class Iterator2d : public SOM::IteratorBase
	{
	public:
		Iterator2d(SOM2d *base, int x, int y) : IteratorBase(),
			base(base), x(x), y(y) {}
		~Iterator2d() {}

		void operator++();
		Neuron &operator*() { return base->neurons[y][x]; }
		Iterator2d *clone() { return new Iterator2d(*this); }

		void assignWith(const IteratorBase *source) {
			// types have been checked by caller
			const Iterator2d *s = static_cast<const Iterator2d *>(source);
			*this = *s;
		}

		// returns the coordinates of the neurons in the multi_img
		// that is created by export_2d
		cv::Point get2dCoordinates() { return cv::Point(x, y); }

		cv::Point3d get3dPos() { return cv::Point3d(x, y, 0); }

		SOM::neighbourIterator neighboursBegin(double radius);
		SOM::neighbourIterator neighboursEnd(double radius);

		// sollten eig nur in SOM2d per friend-klasse zugreifbar sein...
		inline cv::Point getId() const { return cv::Point(x, y); }

	protected:
		bool equal(const SOM::IteratorBase &other) const;

	private:
		SOM2d *base;
		int x, y;
	};

	// Iterates over the neighbours of a neuron within a given radius,
	// in no particular order
	class NeighbourIterator2d : public SOM::NeighbourIteratorBase
	{
	public:
		NeighbourIterator2d(SOM2d *base,
							cv::Point neuron,
							double radius, bool end = false);
		~NeighbourIterator2d() {}

		void operator++();
		Neuron &operator*() { return base->neurons[y][x]; }
		NeighbourIterator2d *clone() { return new NeighbourIterator2d(*this); }

		double getDistance() const;
		double getDistanceSquared() const;
		double getFakeGaussianWeight(double sigma) const;

	protected:
		bool equal(const SOM::NeighbourIteratorBase &other) const;

	private:
		SOM2d *const base;
		const int fromX, toX, toY;
		int x, y;
		const cv::Point neuron;
		const double radiusSquared;
	};

	class Cache2d : public Cache
	{
	public:
                struct PreloadTBB
                {
                        PreloadTBB(const multi_img &image, SOM2d *som,
                                   std::vector<std::vector<cv::Point> > &data)
                                   : image(image), som(som), data(data) { }

                        void operator()(const tbb::blocked_range<int>& r) const;

                        const multi_img &image;
                        SOM2d *const som;
                        std::vector<std::vector<cv::Point> > &data;
                };

		/**
		  @arg som Already trained Self-Organizing map
		  @arg img_height height of the pixel cache (image size, not som size)
		  @arg img_width width of the pixel cache
		  */
		Cache2d(SOM2d *som, int img_height, int img_width) : Cache(), som(som),
			data(std::vector<std::vector<cv::Point> >(img_height,
				std::vector<cv::Point>(img_width, cv::Point(-1, -1)))) { }

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
		SOM2d *const som;
		std::vector<std::vector<cv::Point> > data;
	};

public:
	typedef std::vector<Neuron> Row;
	typedef std::vector<Row> Field;
	friend class Cache2d;

	SOM2d(const vole::EdgeDetectionConfig &conf, int dimension,
		  std::vector<multi_img_base::BandDesc> meta);
	SOM2d(const vole::EdgeDetectionConfig &conf, const multi_img &data,
		  std::vector<multi_img_base::BandDesc> meta);

	/**
	  @arg img_height height of the pixel cache (image size, not som size)
	  @arg img_width width of the pixel cache
	  */
	Cache2d *createCache(int img_height, int img_width);

	int size() const { return width * height; }

	int get2dWidth() const { return width; }
	int get2dHeight() const { return height; }

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
	static double getDistance(const cv::Point &p1, const cv::Point &p2);
	static double getDistance(const cv::Point2d &p1, const cv::Point2d &p2);
	static double getDistanceSquared(const cv::Point &p1, const cv::Point &p2);
	static double getDistanceSquared(const cv::Point2d &p1, const cv::Point2d &p2);

	Field neurons;				///< Neurons in the SOM grid

private:
	const int width;			///< Width  of SOM grid
	const int height;			///< Height of SOM grid
};

#endif // SOM2D_H
