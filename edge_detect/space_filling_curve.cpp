#include <iostream>
#include "highgui.h" // opencv

#include "space_filling_curve.h"

void SpaceFillingCurve::buildHilbert() {

	// initialization values: u r d
	Iteration start; start.push_back('u'); start.push_back('r');  start.push_back('d');
	m_path.push_back(start);

	for(int i = 0; i < m_order-1; i++) {
			Iteration first;
			Iteration second;
			Iteration third;
			Iteration fourth;

		// Take 4 copies of the n-th built to generate n+1-th
		// Modify those copies by mirroring on different axis
		// which is done by function t() and r()
		if(m_saveMemory) {
			first = m_path[0].t();
			second = m_path[0];
			third = m_path[0];
			fourth = m_path[0].r();
			
			m_path.clear();
		} else {
			first = m_path[i].t();
			second = m_path[i];
			third = m_path[i];
			fourth = m_path[i].r();
		}

		// next iteration: take the 4 modified parts and 
		// plug them together using the initialization values u r d
		Iteration next = first + 'u' + second + 'r' + third + 'd' + fourth;
		m_path.push_back(next);
	}
}

void SpaceFillingCurve::buildPeano() {

	// initialization values: u u r d d r u u
	Iteration first; 
	first.push_back('u'); first.push_back('u');  first.push_back('r');
	first.push_back('d'); first.push_back('d');  first.push_back('r');
	first.push_back('u'); first.push_back('u');

	m_path.clear();
	m_path.push_back(first);

	for(int i = 0; i < m_order-1; i++) {

		Iteration n1;
		Iteration n2;
		Iteration n3;

		Iteration n4;
		Iteration n5;
		Iteration n6;

		Iteration n7;
		Iteration n8;
		Iteration n9;

	// take 9 copies of the n-th built to generate n+1-th
	// modify those copies by mirroring on different axis
	// which is done by functions q(), s(), rp(), q()
	if(m_saveMemory) {
		n1 = m_path[0];
		n2 = m_path[0].q();
		n3 = m_path[0];
		n4 = m_path[0].s();
		n5 = m_path[0].rp();
		n6 = m_path[0].s();
		n7 = m_path[0];
		n8 = m_path[0].q();
		n9 = m_path[0];
		
		m_path.clear();
	} else {
		n1 = m_path[i];
		n2 = m_path[i].q();
		n3 = m_path[i];
		n4 = m_path[i].s();
		n5 = m_path[i].rp();
		n6 = m_path[i].s();
		n7 = m_path[i];
		n8 = m_path[i].q();
		n9 = m_path[i];
	}

	// next iteration: take the 9 modified parts and 
	// plug them together using the initialization values u u ...
	Iteration next = n1 + 'u' + n2 + 'u' + n3 + 'r' + n4 + 'd' + n5 + 'd' + n6 + 'r' + n7 + 'u' + n8 + 'u' + n9;

	m_path.push_back(next);

	}
}

void SpaceFillingCurve::generateIndices() {

	unsigned int x = 0;
	unsigned int y = 0;
	
	int iteration = m_saveMemory ?  0 : m_order-1;
	
	// creating the one-dimensional ranking by following the previously
	// generated path commandos
	for(unsigned int index = 0; index < nIndices-1; index++) {

		m_index(y,x) = index;
		
		// following the directions
		switch(m_path[iteration][index]) {
			case 'u': y++; break;
			case 'l': x--; break;
			case 'r': x++; break;
			case 'd': y--; break;
		}
	}
	// assigning the current rank
	m_index(y,x) = nIndices-1;
}

void SpaceFillingCurve::visualizeMatrix() {

	// tranform index matrix into float matrix for the purpose of visualization
	cv::Mat_<float> peanoMatrix(sideLength, sideLength);
	
	for(unsigned int y = 0; y < sideLength; y++) {
		for(unsigned int x = 0; x < sideLength; x++) {
			peanoMatrix(y,x) = m_index(y,x) / (float)nIndices;
		}
	}

	// some image conversion to create a grayscale rank representation
	// on which colored lines will be assigned to visualize its traverse
	std::vector<cv::Mat> color;
	cv::Mat show;
	color.push_back(peanoMatrix);
	color.push_back(peanoMatrix);
	color.push_back(peanoMatrix);
	cv::merge(color, peanoMatrix);
	cv::cvtColor(peanoMatrix, show, CV_BGR2BGRA); 

	// showing the grayscale visualization
	std::cout << "Press any key to traverse the SOM!" << std::endl;
	cv::namedWindow("Scan", 0);
	cv::imshow("Scan",show);
	cv::waitKey();

	unsigned int x = 0;
	unsigned int y = 0;

	cv::Point pOld(0,0);
	cv::Point pNew(0,0);

	// Traverse the rank matrix in ascending order
	for(unsigned int index = 0; index < nIndices-1; index++) {

		pNew.x = x; pNew.y = y;
		
		cv::line(show, pOld, pNew, cv::Scalar(0.,0.2,1.,0.3), 1, CV_AA  );
		pOld.x = pNew.x; pOld.y = pNew.y;
		// showing the grayscale visualization of the rank matrix 
		// overlayed by lines that show the path of the traverse
		cv::imshow("Scan", show);
		cv::waitKey(20); ///< waiting time before next step of traverse

		int iteration = m_saveMemory ?  0 : m_order-1;
		
		switch(m_path[iteration][index]) {
			case 'u': y++; break;
			case 'l': x--; break;
			case 'r': x++; break;
			case 'd': y--; break;
		}
	}
}



