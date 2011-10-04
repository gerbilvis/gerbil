//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//


#ifndef GTM_MISC_H
#define GTM_MISC_H

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "multi_img.h"
#include "cv.h"
#include "highgui.h"


	/**
	* \brief Calculates squared distance between two matrices
	*
	* Calculates squared distance between two matrices.
	* If x has dimension M x N and c has dimension L x N,
	* dest will have dimension M x L
	*
	* @param	x 				Matrix A
	* @param	c 				Matrix B
	* @param	dest 			A * B
	*
	*/
	void dist2(cv::Mat1d x,cv::Mat1d c, cv::Mat1d &dest);
	
	/**
	* \brief Scales eigenvectors by eigenvalues
	*
	*	@param eigenvectors		Matrix containing the eigenvectors
	*	@param eigenvalues		Matrix containing the eigenvalues
	*	@param dest						Matrix containing the scaled eigenvectors	
	*	@param dim_latent			Dimensionality of latent variables
	*
	*/
	void scaleEVec(cv::Mat1d &eigenvectors, cv::Mat1d &eigenvalues,cv::Mat1d &dest, unsigned int dim_latent);
	
	/**
	*	\brief Removes mean and standard deviation of X
	*
	*	@param X			Matrix to be normalized
	*	@param normX	Normalized matrix
	*
	*/
	void normalize(cv::Mat1d &X ,cv::Mat1d &normX);
	
	/**
	*	\brief Create a diagonal matrix with specified values on the diagonal
	*
	*	@param dest				dimension x dimension diagonal matrix
	*	@param dimension	Dimension of square matrix
	*	@param value			Value of the diagonal
	*
	*/	
	void diag(cv::Mat1d &dest , unsigned int dimension, double value );
	
	/**
	*	\brief Read matrix data from a file to cv::Mat structure
	*
	*	Reads a matrix stored in MATLAB format from file.
	*	Matrix elements have to be separated with '\t'
	*
	*/
	void loadFromFile(cv::Mat1d &dest, const char* fn);
	
	//! Converts a multi_img into a cv::Mat structure
	void msi2Mat(multi_img &img, cv::Mat1d &dest);

	//! Split std::string 'str' at substring 'delimiters'
	std::vector<std::string> tokenize(const std::string& str,const std::string& delimiters);
	
	/**
	* \brief Returns elementwise maximum of a vector and a scalar
	*
	*	@param src1 	Input vector
	*	@param src2 	Input scalar
	*	@return 			Vector, that contains elementwise maximum of src1 and src2
	*
	*/
	cv::Mat1d maxVS(cv::Mat1d &src1, double src2);

	/**
	* \brief Finds value and position of maximum element in vector
	*
	*	@param	src		Input vector 
	*	@param	val		Maximum elements
	*	@param	pos		Position of maximum element
	*
	*/
	void maxVal(cv::Mat1d &src, double &val, unsigned int &pos );
	
	/**
	* \brief Column-wise	summation of a matrix
	*
	*	@param src		Matrix to be summed
	*	@param dest		Row vector containing the sum of each column of src
	*
	*/
	void sumCol(cv::Mat1d &src, cv::Mat1d &dest);

	/**
	* \brief Row-wise	summation of a matrix
	*
	*	@param src		Matrix to be summed
	*	@param dest		Column vector containing the sum of each row of src
	*
	*/	
	void sumRow(cv::Mat1d &src, cv::Mat1d &dest);
	
	/**
	* \brief Reshape elements of a vector into a matrix
	*
	*	@param src			Input vector
	*	@param dest			Ouput matrix
	*	@param rows			Rows of output matrix
	*	@param cols			Columns of output matrix
	*	@param rowInput	If true, input vector is expected to be a row vector, otherwise input is expected to be a column vector
	*
	*/
	void reshapeVM(cv::Mat1d &src, cv::Mat1d &dest, int rows, int cols, bool rowInput);
	
	/**
	* \brief Return squared diagonal matrix with diagonal values taken from input vector
	*
	*	@param src	Input vector
	*	@return 		Ouput square diagonal matrix, where the elements on the diagonal correspond to the elements in src
	*
	*/	
	cv::Mat1d diagV(cv::Mat1d &src);
	
	//! Multiplication of a martix src1 and a diagonal matrix src2, that is represented as a vector
	cv::Mat1d sparseMul(cv::Mat1d &src1, cv::Mat1d &src2);
			
	/**
	* \brief Reshape contents of NxMxD matrix to N MxD matrices
	*
	*	@param ndim		D dimensional vector containing N x M matrices
	*	@param oneDim	M x D Matrix 
	*	@param nout		D
	*	@param row		Reshape row-th row of all dimensions of ndim into M x D dimensional matrix
	*
	*/
	void squeeze(std::vector<cv::Mat1d> &ndim, cv::Mat1d &oneDim, unsigned int nout, unsigned int row);
	
	//! Write means to file
	void writeMeans(cv::Mat1d &means, cv::Mat1d &labels, const char *fn, std::ios_base::openmode mode);
	//! Write modes to file
	void writeModes(cv::Mat1d &means, cv::Mat1i &indexMaxR, cv::Mat1d &labels, const char *fn, std::ios_base::openmode mode);
	//! Write means and modes to file
	void writeMeansModes(cv::Mat1d &means,cv::Mat1d &modes, cv::Mat1d &labels, const char *fn, std::ios_base::openmode mode);
	//! Write posterior probability data to text file
	void writePosterior(cv::Mat1d &X, cv::Mat1d &Y,cv::Mat1d post, const char *fn, std::ios_base::openmode mode);
	//! Write posterior probability data as png
	void writePosterior(cv::Mat1d &Responsibilities,cv::Size latentGrid, unsigned int n,cv::Mat1d &postN ,unsigned int iter, std::string fn);
	//! Write magifications data to text file
	void writeMagnifications(cv::Mat1d &X, cv::Mat1d &Y,cv::Mat1d &mag, const char *fn, std::ios_base::openmode mode);
	//! Write results of error function to text file
	std::string writeError(cv::Mat1d &errlog, const char *fn, std::ios_base::openmode mode);
	
	/**
	*	\brief	Draw (randomly) samples from the input image
	*
	*	@param src							N x D dimensional input matrix, where each row contains a multispectral vector
	*	@param samplePercentage	Percentage of drawn samples
	*	@param dest							P x D dimensional matrix, where P is the amount of drawn samples accoring to samplePercentage
	*	@param imgDimension			Size information of multispectral image
	*	@param fixedSeed				If true, random generator is initalized with a fixed seed
	*
	*/
	void drawSamples(cv::Mat1d &src, double samplePercentage, cv::Mat1d &dest,cv::Size imgDimension, bool fixedSeed=false);
	//! Computes Euclidean distance of 2 2-dimensional points
	double vectorDistance(const cv::Point2d &p1, const cv::Point2d &p2);
	//! Computes Euclidean distance of 2 3-dimensional points
	double vectorDistance3d(const cv::Point3d &p1, const cv::Point3d &p2);
	
	/** 
	* \brief Map non-integer values to integer grid points
	*
	*	Performs mapping in to steps:
	*	1. Non-integer latent vaiable positions (lying in the domain ([-1,-1],[1,1]) ) to integer values ([0,0],[N-1.N-1])
	* 2. Map non-integer values from point to nearest integer gridpoint calculated in 1.
	*
	*	@param	point	2-dimensional (non-integer) point
	*	@param	latentX	Non-integer latent x positions
	*	@param	latentY	Non-integer latent y positions
	*	@param	row			Row of mapped integer grid point
	*	@param	col			Column of mapped integer grid point
	*
	*/
	void mapToNearestGridPoint(cv::Point2d &point, cv::Mat1d &latentX, cv::Mat1d &latentY, unsigned int &row, unsigned int &col);
	
#endif
