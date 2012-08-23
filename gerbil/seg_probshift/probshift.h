#ifndef PROBSHIFT_H
#define PROBSHIFT_H

#include "probshift_config.h"
#include "progress_observer.h"
#include <multi_img.h>
#include <cv.h>
#include <lsh.h>
#include <slepceps.h>
#include <petscmat.h>

// work around namespace collisions between PETSc and OpenCV
typedef struct _p_Mat* PetscMat;
typedef struct _p_Vec* PetscVec;


namespace vole {

class ProbShift {
public:
	ProbShift(const ProbShiftConfig& config) : config(config) {}

	cv::Mat1s execute(const multi_img& input, const std::string& base, ProgressObserver *progress = NULL);

private:
	typedef multi_img::Pixel Pixel;
	typedef multi_img::Value Value;

	const ProbShiftConfig &config;

	ProgressObserver *progressObserver;

	// update progress if an observer exists
	bool progressUpdate(int percent);

	// assign labels, fill output mask
	cv::Mat1s assignLabels(const vector<unsigned int> &armode, int width, int height);

	// L2-normalize rows in a PetscMat
	void petscL2normalize(PetscMat &m);

	// perform cluster identification by spectral clustering
	void identifySpectral(PetscMat &petsc_p, vector<unsigned int> &armode, bool destroyP = true);

	// run transition probability matrix power iterations using PETSc
	void powerIterationsPetsc(PetscMat &p, PetscMat &convp, vector<unsigned int> &armode);

	// evaluate accuracy of preallocation for PETSc matrix
	void petscPreallocQuality(const PetscMat &m, double nzfact);

	// perform sign test at certain signficance value alpha
	bool signtest(vector<double> &vals, double alpha, int tail = -1);

	// merge linked modes
	void joinModes(vector<unsigned int> &armode, const vector<int> &modes, const cv::Mat1s &labels);

	// perform mean-shift post-processing on given modes
	vector<unsigned int> meanShiftPostProcess(const vector<unsigned int> &armode, const multi_img &input);

	// convert a square, stlvector-based crf matrix to PetscMat, with optional index remapping
	// (input matrix will be cleared in the process)
	void vectorCrfToPetscMat(vector< vector< pair<int, double> > > &crf, PetscMat &petsc, vector<unsigned int> *map = NULL);

	// convert std::vector<double> to PetscVec
	PetscVec vectorToPetscVec(vector<double> &input);

	// determine centroids of given clusters
	vector<Pixel> getClusterCenters(const vector<int> &modes, const vector<unsigned int> &armode, const multi_img &input);


	// save armode array to plain text file (for development purposes)
	void saveModes(const vector<unsigned int> &armode, std::string filename);

	// load previously stored armode array from file (for development purposes)
	vector<unsigned int> loadModes(std::string filename);

	// helper predicate to sort a crf row by index
	struct CrfPred {
		bool operator()(const pair<int,double> &a, const pair<int,double> &b) {
			return a.first < b.first;
		}
	};

	// structure for bucket sorting of neighborhoods
	struct BsItem {
		/// direction of this point relative to center
		cv::Mat_<double> dirVec;
		/// global index of this point
		unsigned int index;
		/// distance
		double radius;

		BsItem(cv::Mat_<double> dirVec, unsigned int index, double radius) :
		dirVec(dirVec), index(index), radius(radius) {}
	};

	/// write L2 norm into dist, return true if it's less than cmp_dist
	inline bool distL2(const vector<Value> &a, const vector<Value> &b, double cmp_dist, double &dist) const
	{
		dist = 0;
		for (unsigned int i = 0; i < a.size(); ++i)
			dist += (a[i] - b[i])*(a[i] - b[i]);
		dist = std::sqrt(dist);
		return dist < cmp_dist;
	}
};

}

#endif // PROBSHIFT_H
