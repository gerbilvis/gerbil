#ifndef SPACE_FILLING_CURVE_H
#define SPACE_FILLING_CURVE_H

// std library
#include <vector>
// opencv
#include "cv.h"


class SpaceFillingCurve {

	public:

		enum Type { HILBERT, PEANO };
		
		/**
		* SpaceFillingCurve constructor
		* Creates either a Peano or a Hilbert curve of the given order.
		* The order implicitly determines the resulting quadratic rank matrix.
		*
		* This is done iteratively and the internal variable m_saveMemory
		* controls if previous iterations should be kept available or not.
		*
		* @param typ	Type of the Curve: HILBERT or PEANO
		* @param order	Order of the space filling curve
		*/
		SpaceFillingCurve(Type typ, int order) : m_order(order), m_saveMemory(true) { 
			
			if(typ == HILBERT) {
				nIndices = static_cast<unsigned int>(pow(4., m_order));
				sideLength = static_cast<unsigned int>(pow(2., m_order));
				m_index = cv::Mat_<unsigned int>::zeros( sideLength, sideLength);
				
				buildHilbert(); 
			} else {
				nIndices = static_cast<unsigned int>(pow(9., m_order));
				sideLength = static_cast<unsigned int>(pow(3., m_order));
				m_index = cv::Mat_<unsigned int>::zeros( sideLength, sideLength);

				buildPeano();
			}

			generateIndices();
		}

		/**
		* Returns the rank matrix containing the linearized rank
		* for each two dimensional coordinate.
		*
		* @return Reference of the rank matrix
		*/
		cv::Mat_<unsigned int>& getRankMatrix() { return m_index; }
		
		/**
		* Visualizes the Linearization through a grayscale image
		* where intensity values represent the ordering.
		*
		* Additionally visualizes how this matrix will be traversed.
		*/
		void visualizeMatrix();
		
		unsigned int nIndices;		///< number of ranks
		unsigned int sideLength;	///< size of width and height of the rank matrix


	protected:

		
		/**
		* Iteratively applies the rules,grammar for creating a hilbert curve
		* and finally ends up with the desired order of the curve.
		*/
		void buildHilbert();

			
		/**
		* Iteratively applies the rules,grammar for creating a peano curve
		* and finally ends up with the desired order of the curve.
		*/
		void buildPeano();

		/**
		* Creating the linearization by following the previously
		* generated path commandos.
		* Results in having a rank for each position of quadratic matrix.
		*/
		void generateIndices();

		struct Iteration;

		std::vector<Iteration> m_path;	///< vector with each step of the recursion
		cv::Mat_<unsigned int> m_index; ///< rank matrix
		int m_order;					///< order of space filling curve
		bool m_saveMemory;				///< if true only last recursion will be kept

		/**
		* Basic data structure that define each step of an iteration.
		*
		* Its superclass is a std::vector so Iteration can contain
		* unlimited values.
		* Each value is an unsigned char that represents one of four possible
		* directions.
		* There are also some functions defined that make the modification of an
		* iteration pretty comfortable.
		*/
		struct Iteration : public std::vector<unsigned char> {

			Iteration() : std::vector<unsigned char>(0) {}
			Iteration( std::vector<unsigned char> &a) : std::vector<unsigned char>(a) {}


			Iteration operator+( Iteration const& b ) { 
				for(unsigned int i = 0; i < b.size(); i++)
					(*this).push_back( b[i] );

				return (*this);
			}

			Iteration operator+( const char b ) { 
					(*this).push_back( b );

				return (*this);
			}
			
			Iteration t() {
				Iteration tmp(*this);
				for(unsigned int i = 0; i < (*this).size(); i++)
					tmp[i] = t((*this)[i]);

				return tmp;
			}

			Iteration r() {
				Iteration tmp(*this);
				for(unsigned int i = 0; i < (*this).size(); i++)
					tmp[i] = r((*this)[i]);

				return tmp;
			}

			Iteration q() {
				Iteration tmp(*this);
				for(unsigned int i = 0; i < (*this).size(); i++)
					tmp[i] = q((*this)[i]);

				return tmp;
			}

			Iteration rp() {
				Iteration tmp(*this);
				for(unsigned int i = 0; i < (*this).size(); i++)
					tmp[i] = rp((*this)[i]);

				return tmp;
			}

			Iteration s() {
				Iteration tmp(*this);
				for(unsigned int i = 0; i < (*this).size(); i++)
					tmp[i] = s((*this)[i]);

				return tmp;
			}

			unsigned char t(unsigned char c) {
				switch(c) {
					case 'u': return 'r'; 
					case 'l': return 'd';
					case 'd': return 'l';
					case 'r': return 'u';
					default:  return 'e';
				}
			}

			
			unsigned char r(unsigned char c) {
				switch(c) {
					case 'u': return 'l'; 
					case 'r': return 'd';
					case 'd': return 'r';
					case 'l': return 'u';
					default:  return 'e';
				}
			}

			unsigned char q(unsigned char c) {
				switch(c) {
					case 'u': return 'u'; 
					case 'l': return 'r';
					case 'd': return 'd';
					case 'r': return 'l';
					default:  return 'e';
				}
			}

			unsigned char rp(unsigned char c) {
				switch(c) {
					case 'u': return 'd'; 
					case 'l': return 'r';
					case 'd': return 'u';
					case 'r': return 'l';
					default:  return 'e';
				}
			}

			unsigned char s(unsigned char c) {
				switch(c) {
					case 'u': return 'd'; 
					case 'l': return 'l';
					case 'd': return 'u';
					case 'r': return 'r';
					default:  return 'e';
				}
			}
		};
		                                  
};


#endif // SPACE_FILLING_CURVE_H
