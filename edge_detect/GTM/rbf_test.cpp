#include "rbf_test.h"
#include <iostream>
#include <string>
#include "multi_img.h"
#include "gtm_defs.h"

#include <string>

RBF_Test::RBF_Test(const EdgeDetectionConfig *conf)
{
  m_conf = conf;
}
RBF_Test::~RBF_Test()
{}

void RBF_Test::execute()
{

	
	const char* fn_data = "/home/knorx/UNI/Studienarbeit/working_directory/code/reflectance/ms_edges/GTM/data/data_4D.txt";
	
	
	
// 	return ;
	
//   unsigned int ndata = 300;
	unsigned int vector_dim = 4;
	
	FOptions options = { 0, 0.0001, 0.0001, 0.000001, "", "" ,0, 1.0, 0, 30, 30, 30, 30, 30,  0, 0.0001, 0.1, 0.5  };

	cv::Mat_<multi_img::Value> randValues( 1, vector_dim);
	cv::RNG rng(cv::getTickCount());
	rng.fill(randValues, cv::RNG::UNIFORM, cv::Scalar(0.0), cv::Scalar(1.0) );
	
//TODO this is the case fur using actual multi_img data
// 	multi_img::Pixel **samples;
	cv::Mat1d data;
	loadFromFile(data, fn_data);
	cv::Mat1d labels = cv::Mat::ones(data.rows,1, CV_64F);

	//generate labels for example data
	// label only for gnuplot color information: 1 = red, 3 = blue
	for(int x = 0; x < data.rows; x++)
	{	
		if(x > 157 )labels[x][0] = 3.0;
		
	}	
	
	
// 	samples = new multi_img::Pixel *[ndata];
// 	for(unsigned int y = 0; y < ndata; y++) 
// 	{
// 		//shuffle random numbers for each input vector
// 		cv::RNG rng(cv::getTickCount());
// 		rng.fill(randValues, cv::RNG::UNIFORM, cv::Scalar(0.0), cv::Scalar(1.0) );		
// 			// create pixels on heap
// 		samples[y] = new multi_img::Pixel(vector_dim);
// 		multi_img::Pixel *p = samples[y];
// 		for(unsigned int i = 0 ; i < vector_dim;i++)
// 		{
// 			(*p)[i] = randValues[0][i];
// 
// 		}	
// 
// 	}  
// 	
// 	pixel2CVMat(samples, ndata, vector_dim, data);
		
      
  unsigned int nin,nhidden,nout;
  double prior = 0.1;
	
  nin = 2;
  nhidden = 16;
  nout = 4;
  
  S_GTMNet gtmnet;
  
  GTM gtm(m_conf, nin,225,4,nhidden, "GAUSSIAN",prior ,gtmnet);
// 	gtm.setMSIDim(img.width, img.height);
	
	const cv::Size latent_shape = cv::Size(15,15);
	const cv::Size rbf_shape = cv::Size(4,4);
	
	options.display_parameter = 0;
	options.algorithm_line_search = 1;
	
	gtm.gtm_init(gtmnet, options, 1.0, data, "REGULAR", latent_shape, rbf_shape );
	
	options.max_number_function_eval = 30;
	options.display_parameter = 1;
	
	//EM-Algorithm
	EM gtm_em(gtmnet, data, options);
	
	cv::Mat1d means, modes;
	cv::Mat1i indexMaxR;
	gtm.gtmlmean(gtmnet, data, means);
	gtm.gtmlmode(gtmnet, data, modes, indexMaxR);
	
	std::string s_means 			=	m_conf->output_dir + std::string("Means.dat");
	std::string s_modes 			= m_conf->output_dir + std::string("Modes.dat");
	std::string s_meansModes 	= m_conf->output_dir + std::string("meansModes.dat");
	const char* fn_means = s_means.c_str();
	const char* fn_modes = s_modes.c_str();
	const char* fn_meansModes = s_meansModes.c_str();
	writeMeans(means, labels, fn_means, std::ofstream::trunc);
	writeModes(modes, labels, fn_modes, std::ofstream::trunc);
	
	//Join up means and modes
	writeMeansModes(means, modes, labels, fn_meansModes, std::ofstream::trunc);
	
	
	
	
	//Display posterior for a data point
	//Choose an interesting on with a large distance between mean and mode
	cv::Mat1d largeDistance = (means - modes);
	
	cv::Mat1d poweredDistance, summedPoweredDistance;
	cv::pow(largeDistance,2.0, poweredDistance);
	double val;
	unsigned int pos;
	
	sumRow(poweredDistance, summedPoweredDistance);

//TODO returns 251 instead of 250
	maxVal(summedPoweredDistance, val, pos);

	//reshape indixes of latent variables from 1x nlatent to latent_x x latent_y
	cv::Mat1d reshaped, XL,YL;
	cv::Mat1d t1,t2;
	t1 = gtmnet.X.col(0);
	
	reshapeVM((t1), XL, latent_shape.height,latent_shape.width, false);
	
	t2 = gtmnet.X.col(1);
	reshapeVM((t2), YL, latent_shape.height,latent_shape.width, false);
	
	
	//calculate responsibility
	cv::Mat1d resp, act, respRow;
	//fetch data point
	getRow(data, respRow, pos);
	//calculate posterior vector = responisibility for each latent variable to have generated this data point
	gtm.gtmpost(gtmnet, respRow, resp, act);
		

	//reshape responibility vector to latent variable grid
	reshapeVM(resp, reshaped, latent_shape.height,latent_shape.width, true);

	//calculate index of latent variable htat hast generated data point with maximum posterior probability
	cv::Point p = maxPosterior(reshaped);
	std::cout << p.y << " " <<p.x <<std::endl;
	
	
// 	reshaped /= val;
// 	std::cout << reshaped <<std::endl;
	std::string post = m_conf->output_dir + std::string("post1.dat");
	const char* fn_post = post.c_str();
	writePosterior(t1,t2, reshaped, fn_post, std::ofstream::trunc);
	
	cv::Mat showResp;
	reshaped.convertTo( showResp, CV_8UC1, 255.);
	cv::imwrite(m_conf->output_dir+m_conf->msi_name + "_"  + "_posterior.png", showResp);
	
	cv::Mat1d magnifications, Mags;
	gtm.gtmmag(gtmnet, magnifications);
	
	reshapeVM(magnifications, Mags, latent_shape.width,  latent_shape.height, false);

	
	cv::Mat showMag;
	double max = 0.0;
	for(int y = 0; y < latent_shape.height; y++)
	{
		double *rowptr = Mags[y];
		for(int x = 0; x < latent_shape.width; x++)
		{	
			if(rowptr[x] >max)max = rowptr[x];
		}	
	}	
	Mags /= max;
	Mags.convertTo( showMag, CV_8UC1, 255.);
	cv::imwrite(m_conf->output_dir+m_conf->msi_name + "_"  + "_mag.png", showMag);
	
	
}


