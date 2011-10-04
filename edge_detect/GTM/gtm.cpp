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



#include <iostream> 
#include <cassert> 
#include <string>
#include <sstream>
#include "rbfnet.h"
#include "gtm.h"

#include "cv.h"
#include <talkingwatch.h>
#include <stopwatch.h>



GTM::GTM(const EdgeDetectionConfig *conf, FOptions &options, unsigned int dim_latent, unsigned int nlatent, unsigned int dim_data, unsigned int ncentres, std::string rbffunc, double prior,S_GTMNet &net)
	: m_conf(conf),m_options(options)
{
//   net.type = "gtm";
//   net.nin = dim_data;
//   net.dim_latent = dim_latent;
//   
//   
//   RBFNet *rbf = new RBFNet(dim_latent,ncentres,dim_data, rbffunc, prior ,net.rbfnet);
//   rbf->rbfprior(rbffunc, dim_latent, ncentres, dim_data, net.rbfnet);
//   
//   net.rbf = rbf;
//   
//   GMM *gmm = new GMM(dim_data, nlatent , "SPHERICAL", net.gmmnet);
//   net.gmm = gmm;
// 	net.gtm = this;

	m_latentWidth = UINT_MAX;
	m_latentHeight = UINT_MAX;
	
	//initalize random seed
	srand ( time(NULL) );

}

GTM::GTM(const EdgeDetectionConfig *config, multi_img *img )
{
	m_conf = config;
	m_msi = img;
	
	m_latentWidth = UINT_MAX;
	m_latentHeight = UINT_MAX;
	
	//initalize random seed
	srand ( time(NULL) );
	
}

//the main GTM execution process
void GTM::execute()
{
		vole::Talkingwatch w_overall;
		vole::Talkingwatch w_init;
		withGraph = m_conf->graph_withGraph;
		withUMap = m_conf->withUMap;

		unsigned int nin,nhidden,nout,nlatent, data_dim;
		double prior = 0.1;
		FOptions options = { 0, 0.0001, 0.0001, 0.000001, "", "" ,0, 1.0, 0, 30, 30, 30, 30, 30,  0, 0.0001, 0.1, 0.5  };
		options.max_number_function_eval = m_conf->gtm_numIterations;
		nin = 2; // == dim_latent
		nhidden = m_conf->gtm_numRbf * m_conf->gtm_numRbf;
// 		
		nlatent = m_conf->gtm_numLatent * m_conf->gtm_numLatent;
		data_dim = m_msi->size();

		nout = data_dim;

		std::cout <<"Data dim: " <<  data_dim << std::endl;
// 
		cv::Mat1d data; 
		cv::Mat1d randData;

		msi2Mat(*m_msi, data);
// 
		drawSamples(data, m_conf->gtm_samplePercentage, randData, cv::Size(m_msi->width,m_msi->height), m_conf->fixedSeed );

		std::cout << "# Running GTM with " <<std::endl;
		std::cout << "# "<<nlatent << " (" << m_conf->gtm_numLatent << " x " <<m_conf->gtm_numLatent <<") latent variables of dimension "<<nin  <<std::endl;
		std::cout << "# " <<nhidden <<" rbf centres" <<std::endl;
		std::cout << "# Sample dimensionality: " <<data.rows <<" x " << data.cols <<std::endl;
// 
		std::cout << "# Subset data dimensionality: " <<randData.rows <<" x " << randData.cols << " identical to " << (100.0 * m_conf->gtm_samplePercentage)<< " % of input space"<<std::endl;
// 		
// 		GTM gtm(&config,nin,nlatent,data_dim,nhidden, config.gtm_actfn ,prior ,gtmnet);
		m_gtmnet.type = "gtm";
		m_gtmnet.nin = data_dim;
		m_gtmnet.dim_latent = nin;
		
		
		RBFNet *rbf = new RBFNet(m_gtmnet.dim_latent,nhidden,data_dim, m_conf->gtm_actfn, prior ,m_gtmnet.rbfnet);
		rbf->rbfprior(m_conf->gtm_actfn, m_gtmnet.dim_latent, nhidden, data_dim, m_gtmnet.rbfnet);
		m_gtmnet.rbf = rbf;
// 		m_gtmnet.rbf->dumpNet(m_gtmnet.rbfnet);
		
		GMM *gmm = new GMM(data_dim, nlatent , "SPHERICAL", m_gtmnet.gmmnet, m_conf->fixedSeed);
		m_gtmnet.gmm = gmm;
		m_gtmnet.gtm = this;		
		
		m_MSIwidth	= m_msi->width; 
		m_MSIheight	= m_msi->height;
		
		const cv::Size latent_shape = cv::Size(m_conf->gtm_numLatent,m_conf->gtm_numLatent);
		const cv::Size rbf_shape = cv::Size(m_conf->gtm_numRbf,m_conf->gtm_numRbf);
		
		m_options.display_parameter = 0;
		m_options.algorithm_line_search = 1;
		
		std::cout << "# Initializing GTM" << std::endl;

		gtm_init(m_gtmnet, options, 1.0, randData, "REGULAR", latent_shape, rbf_shape );
		
		options.max_number_function_eval = m_conf->gtm_numIterations;
		options.display_parameter = 1;
		
		
		//EM-Algorithm
		std::cout << "# Run EM-Algorithm ("<< m_conf->gtm_numIterations << " iterations)" << std::endl;
		
		infoStream << getFileExtension() <<std::endl;
		infoStream <<  w_init.print("initialisation") <<std::endl;
		vole::Talkingwatch w_em;
		EM gtm_em(m_conf, m_gtmnet, randData, options);
		
		std::string errlog = gtm_em.getErrlog();
		infoStream << std::endl << errlog << std::endl;
		infoStream <<  w_em.print("EM") <<std::endl;
		
		vole::Stopwatch post;
		std::cout << "# Calculate posteriors" << std::endl;

		cv::Mat1d means,modes;
		cv::Mat1d means2, modes2;
		vole::Stopwatch meansFull;
		vole::Talkingwatch w_means;
		gtmlmean(m_gtmnet, data, means);
		meansFull.print("Means Full");
		infoStream << w_means.print("Means");

		cv::Mat1i indexMaxR,indexMaxR2;
		vole::Stopwatch modesFull;		

		vole::Talkingwatch w_modes;
		gtmlmode(m_gtmnet, data, modes, m_maxModes);
		modesFull.print("Modes Full");
		infoStream << w_modes.print("Modes");
		
// 		vole::Stopwatch modesSingle;		
// 		gtmlmode_par(m_gtmnet, data, modes2, indexMaxR2);		
// 		modesSingle.print("Modes Single");
	
		std::string s_means 			=	m_conf->output_dir + getFileExtension() +std::string("_Means.dat");
		std::string s_modes 			= m_conf->output_dir + getFileExtension() +std::string("_Modes.dat");
		std::string s_meansModes 	= m_conf->output_dir + getFileExtension() +std::string("_meansModes.dat");
		const char* fn_means = s_means.c_str();
		const char* fn_modes = s_modes.c_str();
		const char* fn_meansModes = s_meansModes.c_str();
		
		cv::Mat1d labels = cv::Mat::ones(data.rows, 1U, CV_32F);
		
		writeMeans(means, labels, fn_means, std::ofstream::trunc);
		writeModes(modes,m_maxModes, labels, fn_modes, std::ofstream::trunc);
// 		
// 				//Join up means and modes
		writeMeansModes(means, modes, labels, fn_meansModes, std::ofstream::trunc);

		cv::Mat1d t1,t2;
		
		t1 = m_gtmnet.X.col(0);
		reshapeVM((t1), latentXCoords, latent_shape.height,latent_shape.width, false);
		
		t2 = m_gtmnet.X.col(1);
		reshapeVM((t2), latentYCoords, latent_shape.height,latent_shape.width, false);

		post.print("Calculating posteriors");

		vole::Stopwatch mag;
		vole::Talkingwatch w_mag;
		std::cout << "# Calculate magnification factors" << std::endl;
		
		cv::Mat1d magnifications, Mags;
		gtmmag(m_gtmnet, magnifications);
		infoStream << w_mag.print("Magnifications");
		reshapeVM(magnifications, Mags, latent_shape.width,  latent_shape.height, false);

		std::string s_mag 	= m_conf->output_dir + getFileExtension() +std::string("_mag.dat");
		const char* fn_mag = s_mag.c_str();
		writeMagnifications(latentXCoords,latentYCoords,Mags, fn_mag,std::ofstream::trunc);
// return;	//exit for test data
		cv::Mat showMag 		= cv::Mat::zeros(latent_shape.height,latent_shape.width, CV_8UC1);
		cv::Mat1d graphMag 	= cv::Mat::zeros(latent_shape.height,latent_shape.width, CV_64F);
		double max = 0.0;
		double min = DBL_MAX;
		for(int y = 0; y < latent_shape.height; y++)
		{
			double *rowptr = Mags[y];
			for(int x = 0; x < latent_shape.width; x++)
			{	
				if(rowptr[x] >max)max = rowptr[x];
				if(rowptr[x] <min)min = rowptr[x];
			}	
		}	
		
		for(int y = 0; y < latent_shape.height; y++)
		{
			double *graphMagPtr = graphMag[y];	
			uchar *showMagPtr = showMag.ptr<uchar>(y);
			double *magPtr = Mags[y];
			
			for(int x = 0; x < latent_shape.width; x++)
			{	
				showMagPtr[x] = static_cast<uchar>( 255.0 * (magPtr[x] / max));
				graphMagPtr[x] = magPtr[x] / min; //set minimal value to 1, take logarithm and add 1, such that minimal distance is 1
				graphMagPtr[x] *= std::exp(1.0);  // set minimal value to 'e'
				graphMagPtr[x] = std::log(graphMagPtr[x]); // minimal value is again 1, but values are scaled now logarithmic
				
			}	
		}			
		
		cv::imwrite(m_conf->output_dir+ getFileExtension() + "_mag.png", showMag);
		mag.print("Calculating magnification factors");		
			
// 		mapMagnifications(latentMap, graphMag);
		
		if(withUMap || withGraph)
			initDijkstra(Mags);
			
		vole::Stopwatch edge;
		std::cout << "# Calculate edges" << std::endl;

		cv::Mat1d meansX, meansY, modesX, modesY,postX,postY;
	
		//mode :: 0:Sobel, 1: Scharr
// 		getEdge(latentMap, sobelX, sobelY,0);
		vole::Talkingwatch w_edge;

		getEdgeByMeans(means, meansX, meansY,0);
		getEdgeByModes(modes, modesX, modesY,0);
		
		
		cv::Mat meansXShow,meansYShow;
		cv::Mat modesXShow,modesYShow;

		meansX.convertTo( meansXShow, CV_8UC1, 255.);
		meansY.convertTo( meansYShow, CV_8UC1, 255.);
		std::string meansDx = getFileExtension()+ "_gtm_means_X.png";
		std::string meansDy = getFileExtension()+ "_gtm_means_Y.png";
		
		cv::imwrite(m_conf->output_dir+ meansDx, meansXShow);
		cv::imwrite(m_conf->output_dir+ meansDy, meansYShow);

		modesX.convertTo( modesXShow, CV_8UC1, 255.);
		modesY.convertTo( modesYShow, CV_8UC1, 255.);
		
		std::string modesDx = getFileExtension()+ "_gtm_modes_X.png";
		std::string modesDy = getFileExtension()+ "_gtm_modes_Y.png";
		
		cv::imwrite(m_conf->output_dir+ modesDx, modesXShow);
		cv::imwrite(m_conf->output_dir+ modesDy, modesYShow);
		
		infoStream << w_edge.print("Edge generation");
		
		edge.print("Calculate edges");
		std::cout<< "Talking watch: " << w_overall.print("Overall time") <<std::endl;
		infoStream << w_overall.print("Overall time");
		
		std::ofstream outFile;
		std::string name = m_conf->output_dir + getFileExtension() + "_time.txt";
		outFile.open(name.c_str(),std::ios_base::out | std::ios_base::trunc);
		if(!outFile.is_open())
			std::cout << "File not open!" <<std::endl;
		outFile << infoStream.str();
		outFile.close();		

}

void GTM::gtm_init(S_GTMNet &net, FOptions &options, double width_factor, cv::Mat1d &data, std::string samp_type, const cv::Size &latent_grid_width, const cv::Size &rbf_hidden_units )
{
	
  assert(net.type == "gtm");
  
  //only regular implemented here
  assert(samp_type == "REGULAR");
  //dimensionality of latent variables have to be smaller or equal to dimensionality of data 
  assert(net.dim_latent <= (unsigned int)data.cols);
  
  unsigned int nlatent, nhidden;
  
	m_latentWidth = latent_grid_width.width;
	m_latentHeight = latent_grid_width.height;
	
  nlatent = net.gmmnet.ncentres;
  nhidden = net.rbfnet.nhidden;

  cv::Size l_samp_size, rbf_samp_size;
  
  if(samp_type == "REGULAR")
  {
    l_samp_size = latent_grid_width;
    rbf_samp_size = rbf_hidden_units;
    
    assert(( ( l_samp_size.width * l_samp_size.height)== nlatent) && (rbf_samp_size.width * rbf_samp_size.height) == nhidden);
    
    cv::Mat1d samples;
    if(1 == net.dim_latent)
    {
      gtm_rctg(l_samp_size,net.X, net.dim_latent);
      net.rbf->rbfsetfw(net.rbfnet, width_factor);

    }
    else if(2 == net.dim_latent)
    {
      gtm_rctg(l_samp_size,net.X, net.dim_latent);
      gtm_rctg(rbf_samp_size,net.rbfnet.c, net.dim_latent);
      net.rbf->rbfsetfw(net.rbfnet, width_factor);

    }

		//do the PCA for initialization
		cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW);
		cv::Mat1d eVals = pca.eigenvalues;
		

		//eigenvalues are stored as row vectors, we need column vectors
		cv::Mat1d eVecs;
		cv::transpose(pca.eigenvectors, eVecs);
		 
		cv::Mat1d A,normX;
	  //scale principal components by eigenvalues

		scaleEVec(eVecs,eVals,A, net.dim_latent );

		cv::Mat1d outputs,phi,distances;
		net.rbf->rbffwd(net.rbfnet, net.X, outputs, phi, distances);
		
		//Normalise X to ensure 1:1 mapping of variances and calculate weights as solution of 
		normalize(net.X, normX);
	
		cv::Mat1d A_trans;
		cv::transpose(A,A_trans);
		cv::Mat1d tmp = normX *A_trans;
		
		//Phi * W = normX * A
		cv::solve(phi, tmp, net.rbfnet.w2, cv::DECOMP_QR);

		net.rbfnet.b2.setTo(0.0);
		//calculate columnwise mean and set the value as bias
		for(int y = 0; y < data.rows; y++)
		{
			double *rowptr = data[y];
			double *b2_ptr = net.rbfnet.b2[0];
			for(int x = 0; x < data.cols; x++)
			{
				b2_ptr[x] += rowptr[x];
			}	
		}
		net.rbfnet.b2 /= data.rows;
		//Must also set initial value of variances
		//Find average distance between nearest centres
		//Ensure that distance of centre to itself is excluded by setting diagonal entries to DBL_MAX
		cv::Mat1d gmm_output,gmm_phi,gmm_distances;
		
		//creates some bias !!
		net.rbf->rbffwd(net.rbfnet, net.X, gmm_output, gmm_phi,gmm_distances);
		net.gmmnet.centres = gmm_output;
	
		cv::Mat1d d;
		dist2(net.gmmnet.centres,net.gmmnet.centres, d);
		
		cv::Mat1d diagGmmCentres;
		diag(diagGmmCentres, net.gmmnet.ncentres, DBL_MAX);
		
		d += diagGmmCentres;
		cv::Mat1d minVal = cv::Mat::zeros(1, net.gmmnet.ncentres, CV_64F);
		minVal.setTo(DBL_MAX);
		
		for(int y = 0; y < d.rows; y++)
		{
			double *rowptr = d[y];
			double *min_ptr = minVal[0];
			for(int x = 0; x < d.cols; x++)
			{
				if(rowptr[x] < min_ptr[x] ) min_ptr[x] = rowptr[x];
			}	
		}
		cv::Scalar minMean = cv::mean(minVal);
		double sigma = minMean[0]/2.0;
		
		//Now set covariance to minimum of sigma and next largest eigenvalue
		
		//check if latent dimensionality is smaller than data dimensionality
		if(net.dim_latent <  (unsigned int)data.cols)
		{
			double nextEVal = 0.0;
			
			//look for the net.dim_latent+1 eigenvalue
			for(int y = 0; y< eVals.rows;y++)
			{
				double *rowPtr = eVals[y];
				if(y == (net.dim_latent+1))
				{
					nextEVal = rowPtr[0];
					break;
				}	
			}	
			sigma = std::min(sigma, nextEVal); 
		}	
		
		net.gmmnet.covars = sigma * cv::Mat::ones(1,net.gmmnet.ncentres,CV_64F);
		
  } 
  else if(samp_type == "UNIFORM" || samp_type == "GAUSSIAN")
  {
    //not implemented here
  } 
  
}

void GTM::gtmmag(S_GTMNet &net, cv::Mat1d &magnifications)
{
	
	std::vector<cv::Mat1d> jac(net.rbfnet.nout);
	cv::Mat1d tmp,tmp_trans;
	
	net.rbf->rbfjacob(net.rbfnet, net.X, jac);
	
	unsigned int ndata = net.X.rows;
	magnifications = cv::Mat::zeros(ndata, 1U, CV_64F);
	
	for(unsigned int n =0; n < ndata;n++)
	{
		double *rowPtr = magnifications[n];

		squeeze(jac,tmp,net.rbfnet.nout, n );
		
		cv::transpose(tmp, tmp_trans);
		
		rowPtr[0] = std::sqrt( cv::determinant(tmp*tmp_trans) );
	}
	
}

void GTM::gtmpost(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &post, cv::Mat1d &activations)
{
	cv::Mat1d phi, distances;
	net.rbf->rbffwd(net.rbfnet, net.X, net.gmmnet.centres,phi,distances);
	
	net.gmm->gmmpost(net.gmmnet, data, post, activations);
}

void GTM::gtmprob(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &prob)
{
	cv::Mat1d  phi, distance;
	net.rbf->rbffwd(net.rbfnet, net.X, net.gmmnet.centres, phi, distance);
	
	net.gmm->gmmprob(net.gmmnet, data, prob);
	
}

void GTM::gtmlmean(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &means)
{
	cv::Mat1d act, R;
	gtmpost(net, data, R, act);

	means = cv::Mat::zeros(data.rows, net.dim_latent, CV_64F);

	means = R * net.X;
	
}

void GTM::gtmlmean_par(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &means)
{
	cv::Mat1d act, R;
	means = cv::Mat::zeros(data.rows, net.dim_latent, CV_64F);
	cv::Mat1d dataN;
	for(int n = 0; n < data.rows; n++)
	{	
		double *meansPtr = means[n];
		dataN = data.row(n);
		gtmpost(net,dataN , R, act);
	
		cv::Mat1d m1 = R * net.X.col(0); 
		cv::Mat1d m2 = R * net.X.col(1);
		meansPtr[0] = *m1[0];
		meansPtr[1] = *m2[0];

	}	
}

void GTM::gtmlmode(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &modes, cv::Mat1i &indexMaxR)
{
	cv::Mat1d act, R;
	unsigned int ndata = data.rows;
	modes = cv::Mat::zeros(ndata, net.dim_latent, CV_64F);
	gtmpost(net, data, R, act);
	
	//indexMaxR	: index of Latent variable that produced maximum responsibility
	//modes			:	value of latent variables
	gtmmaxresp(net, R, indexMaxR);
// 	gtmmaxmode(net, ndata, indexMaxR, modes);
	gtmmaxmode(net, indexMaxR, modes);
		
}

void GTM::gtmlmode_par(S_GTMNet &net, cv::Mat1d &data, cv::Mat1d &modes, cv::Mat1i &indexMaxR)
{
	cv::Mat1d act, R;
	unsigned int ndata = data.rows;
	cv::Mat1d dataN;
	modes = cv::Mat::zeros(ndata, net.dim_latent, CV_64F);
	indexMaxR = cv::Mat::zeros(data.rows,1,CV_32S);
	for(int n = 0; n < data.rows; n++)
	{
		int * indPtr = indexMaxR[n];
		dataN = data.row(n);
		gtmpost(net, dataN, R, act);
		cv::Point maxResp;
		cv::minMaxLoc(R, NULL, NULL, NULL, &maxResp);
		indPtr[0] = maxResp.x;
	}	
		
	//indexMaxR	: index of Latent variable that produced maximum responsibility
	//modes			:	value of latent variables
	
// 	gtmmaxmode(net, ndata, indexMaxR, modes);
	gtmmaxmode(net, indexMaxR, modes);
		
}

void GTM::gtmmaxresp(S_GTMNet &net, cv::Mat1d &R, cv::Mat1i &maxIndex)
{

	int index;
	int dimR = R.cols;
	maxIndex = cv::Mat::zeros(R.rows,1,CV_32S);
	double val;
	
	for(int y = 0; y < R.rows; y++)
	{
		int *indexPtr = maxIndex[y]; 
		
		index = INT_MAX;
		val = 0.0;
		double *rowPtr = R[y];
		for(int x = 0; x < dimR; x++)
		{
			if(rowPtr[x] > val)
			{
				val = rowPtr[x];
				index = x;
			}	
		}
		//if zero probability, map to RANDOM latent variable  (we actually know nothing about the latent variable 
		//that produced that point, so choose a random one (best solution?!)
		
		if(index == INT_MAX)
		{
			std::cout << "Zero prob" <<std::endl;
			index = (std::rand() % net.gmmnet.ncentres);
		}	
		indexPtr[0] = index;
	}	

}

void GTM::gtmmaxmode(S_GTMNet &net, cv::Mat1i &maxIndex, cv::Mat1d &modes)
{
	// store for every data point the 2-D coordinates of the latent variables belonging to the maximum posterior probability
// 	modes = cv::Mat::zeros(ndata, net.dim_latent, CV_64F);
	unsigned int ndata = maxIndex.rows;
	for(unsigned int y = 0; y < ndata; y++)
	{
		double *rowPtr = modes[y];
		int *indexPtr = maxIndex[y];
		
		for(unsigned int x = 0; x < net.dim_latent; x++)
		{

			double *latentPtr = net.X[indexPtr[0]];
			rowPtr[x] = latentPtr[x];
			
		}
		
	}	
}

void GTM::gtm_rctg(cv::Size latent_samp_size, cv::Mat1d &latent_samp, int dim)
{
  assert(dim <=2);
  
  int xDim = latent_samp_size.width;
  int yDim = latent_samp_size.height;
  
  cv::Mat1d X = cv::Mat::zeros(latent_samp_size, CV_64F);
  cv::Mat1d Y = cv::Mat::zeros(latent_samp_size, CV_64F);
  
  double maxX = 0.0;
  double maxY = 0.0;
  
  if(dim == 1)
  { 
    for(int y = 0; y < yDim; y++)
    {
      double *rowptr = X[y]; 
      for(int x = 0; x < yDim; x++)
      {
        rowptr[x] =  (double)(y);
        if(rowptr[x] > maxX)maxX = rowptr[x];
      }
    }
    
    latent_samp = cv::Mat1d((xDim * yDim), 1, CV_64F);

    for(int y = 0; y < ( yDim * xDim); y++)
    {
      double *rowptr = latent_samp[y];
      double *X_ptr = X[y%(xDim)];
      
      rowptr[0] = X_ptr[y % (yDim )];
        
    } 
    
    for(int i = 0; i < ( xDim * yDim) ; i++)
    { 
      latent_samp[i][0] = 2.0* (latent_samp[i][0] - maxX/2.0 )/maxX;
    }

  }
  else
  { 
    for(int y = 0; y < yDim; y++)
    {
      double *rowptr = X[y]; 
      for(int x = 0; x < xDim; x++)
      {
        rowptr[x] =  (double)x;
        if(rowptr[x] > maxX)maxX = rowptr[x];
      }
    } 

    for(int y = 0; y < yDim; y++)
    {
      double *rowptr = Y[y]; 
      for(int x = 0; x < xDim; x++)
      {
        rowptr[x] =  (double)(yDim- 1 - y);
        if(rowptr[x] > maxY)maxY = rowptr[x];
      }
    }
      
    latent_samp = cv::Mat1d((xDim * yDim), 2, CV_64F);
    unsigned int x_pos = 0;

    for(int y = 0; y < ( yDim * xDim); y++)
    {
      double *rowptr = latent_samp[y];
      double *X_ptr = X[y%(yDim)];
      double *Y_ptr = Y[y%(yDim)];

      for(int x = 0; x < dim ; x++)
      {
        if(x == 0)
          rowptr[x] = X_ptr[x_pos];
        else if(x == 1)
          rowptr[x] = Y_ptr[x % (yDim )];
        
      }
      
      if(((y+1)%(yDim))==0 && y > 0)
      {
        x_pos++;
      } 
        
    } 
    for(int i = 0; i < ( xDim * yDim) ; i++)
    { 
      latent_samp[i][0] = 2 * (latent_samp[i][0] - maxX/2.0 )/maxX;
      latent_samp[i][1] = 2 * (latent_samp[i][1] - maxY/2.0 )/maxY;
    }
  } 
  
}


void GTM::initDijkstra(cv::Mat1d &magnifications)
{
	std::cout << "# Initializing Dijkstra" <<std::endl;
	unsigned int nlatent = m_conf->gtm_numLatent;
	unsigned int nnodes = nlatent * nlatent;
	msi::Configuration graphConf;
	graphConf.output = m_conf->output_dir;
  graphConf.directed = false;
  graphConf.completing = false;
  graphConf.periodic = false;
  graphConf.insertion = msi::UNDEFINED;
  graphConf.nodes = nnodes;
	graphConf.width = m_latentWidth;
	graphConf.height = m_latentHeight;
  graphConf.startcomp = 0;
  graphConf.beta = 0.0;
  graphConf.phi = 0.0;	
  graphConf.finishcomp = graphConf.nodes;
  graphConf.initialDegree = m_conf->sw_initialDegree;
  graphConf.graph_type = "MESH";
	graphConf.maxIter = m_conf->gtm_numIterations;
	
	m_mesh = new msi::Mesh(graphConf, ( nnodes));
	//insert dummy Neurons
	for(unsigned int i = 0; i< nnodes;i++)
	{
		m_mesh->update(i, new Neuron(1));
	}	

	m_mesh->initDijkstra();
	if(withUMap)
		umatrix(magnifications);

}

double GTM::graphDistance(cv::Point2d p1, cv::Point2d p2)
{
	int *fromPtr = m_maxModes[(unsigned int)p1.y * m_latentWidth + (unsigned int)p1.x];
	int *toPtr = m_maxModes[(unsigned int)p2.y * m_latentWidth + (unsigned int)p2.x];

	unsigned int index1 = fromPtr[0];
	unsigned int index2 = toPtr[0];

	if(withUMap)
		return m_mesh->getDistance(index1,index2, true);
	else
		return m_mesh->getDistance(index1,index2, false);
}

double GTM::graphDistance(unsigned int index1Y,unsigned int index1X, unsigned int index2Y,unsigned int index2X, bool useEdgeWeigths)
{
	unsigned int indexA = (index1Y * m_latentWidth) + index1X;
	unsigned int indexB = (index2Y * m_latentWidth) + index2X;
	return graphDistance(indexA, indexB, useEdgeWeigths);
}

double GTM::graphDistance(unsigned int index1, unsigned int index2, bool useEdgeWeigths)
{
	return m_mesh->getDistance(index1, index2,useEdgeWeigths);
}

void GTM::getEdgeByMeans(cv::Mat1d &means, cv::Mat1d &distanceX, cv::Mat1d &distanceY , int mode)
{
	//will contain for every pixel the according 2D position of the mean value of the according latent density
	
	bool withUMap = m_conf->withUMap;
	bool withGraph = m_conf->graph_withGraph;
		
	cv::Mat2d locations(m_MSIheight, m_MSIwidth);
	double c1,c2,c3,fraction;
	c1 =1.0;
	c2 =2.0;
	c3 =1.0;
	fraction = 1.0/(c1+c2+c3);
	
	distanceX = cv::Mat::zeros(m_MSIheight, m_MSIwidth, CV_64F);
	distanceY = cv::Mat::zeros(m_MSIheight, m_MSIwidth, CV_64F);
	
	for (unsigned int y = 0; y < m_MSIheight ; y++)
	{
		
		cv::Vec2d *rowPtr = locations[y];
		for (unsigned int x = 0; x < m_MSIwidth ; x++)
		{
			double *indPtr = means[y * m_MSIwidth + x];
			rowPtr[x][0] = indPtr[0];
			rowPtr[x][1] = indPtr[1];
		}
	}
	
	double maxIntensity =0.0;
	double valx,valy =0.0;
	
	for(unsigned int y = 1; y < (m_MSIheight - 1U); y++)
	{
		double *x_ptr = distanceX[y];
		double *y_ptr = distanceY[y];
		cv::Vec2d *i_ptr = locations[y];
		cv::Vec2d *i_uptr = locations[y-1];
		cv::Vec2d *i_dptr = locations[y+1];
		double xx,yy;
		for(unsigned int x = 1; x < (m_MSIwidth - 1U); x++)
		{
			
			{	// y-direction
				xx = (c1 * i_uptr[x-1][0] + c2 * i_uptr[x][0] + c3 * i_uptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_uptr[x][1] + c3 * i_uptr[x+1][1]) * fraction;
				cv::Point2d u(xx, yy);
				xx = (c1 * i_dptr[x-1][0] + c2 * i_dptr[x][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_dptr[x-1][1] + c2 * i_dptr[x][1] + c3 * i_dptr[x+1][1]) * fraction;
				cv::Point2d d(xx, yy);

				if(withUMap || withGraph)
				{
					unsigned int colU, rowU, colD, rowD, indexU, indexD;
					mapToNearestGridPoint(u, latentXCoords, latentYCoords, rowU, colU );
					mapToNearestGridPoint(d, latentXCoords, latentYCoords, rowD, colD );
					indexU = rowU * latentXCoords.cols + colU;
					indexD = rowD * latentXCoords.cols + colD;
					
					valy = graphDistance(indexU, indexD,withUMap);
				}	
				else	
					valy = vectorDistance(u, d);

				if (maxIntensity < valy)
					maxIntensity = valy;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valy = -valy;
				y_ptr[x] =  valy;
			}
			{	// x-direction
				xx = (c1 * i_uptr[x-1][0] + c2 * i_ptr[x-1][0] + c3 * i_dptr[x-1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_ptr[x-1][1] + c3 * i_dptr[x-1][1]) * fraction;
				cv::Point2d u(xx, yy);
				xx = (c1 * i_uptr[x+1][0] + c2 * i_ptr[x+1][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x+1][1] + c2 * i_ptr[x+1][1] + c3 * i_dptr[x+1][1]) * fraction;
				cv::Point2d d(xx, yy);
				
				if(withUMap || withGraph)
				{
					unsigned int colU, rowU, colD, rowD, indexU, indexD;
					mapToNearestGridPoint(u, latentXCoords, latentYCoords, rowU, colU );
					mapToNearestGridPoint(d, latentXCoords, latentYCoords, rowD, colD );
					indexU = rowU * latentXCoords.cols + colU;
					indexD = rowD * latentXCoords.cols + colD;
					
					valx = graphDistance(indexU, indexD,withUMap);
				}	
				else	
					valx = vectorDistance(u, d);

				if (maxIntensity < valx)
					maxIntensity = valx;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valx = -valx;
				x_ptr[x] = valx;
			}
		}	
	
	}
	//normalization
	for (unsigned int y = 0; y < m_MSIheight ; y++)
  {
    double* x_ptr = distanceX[y];
    double* y_ptr = distanceY[y];
    for(unsigned int x = 0; x < m_MSIwidth ; x++)
    {
      x_ptr[x] =  ((x_ptr[x] + maxIntensity)*0.5/maxIntensity);
      y_ptr[x] =  ((y_ptr[x] + maxIntensity)*0.5/maxIntensity);
    }
  }

}
		
void GTM::getEdgeByModes(cv::Mat1d &modes, cv::Mat1d &distanceX, cv::Mat1d &distanceY , int mode)
{
	
	bool withUMap = m_conf->withUMap;
	bool withGraph = m_conf->graph_withGraph;
	
	double c1,c2,c3,fraction;
	c1 =1.0;
	c2 =2.0;
	c3 =1.0;
	fraction = 1.0/(c1+c2+c3);
		
	distanceX = cv::Mat::zeros(m_MSIheight, m_MSIwidth, CV_64F);
	distanceY = cv::Mat::zeros(m_MSIheight, m_MSIwidth, CV_64F);
	if(withGraph)
		std::cout <<"Graph enabled" <<std::endl;
	
	//will contain for every pixel the according 2D position of the maximum value of the according latent density
	cv::Mat2d locations(m_MSIheight, m_MSIwidth);
	cv::Mat1i indices(m_MSIheight, m_MSIwidth);
	
	for (unsigned int y = 0; y < m_MSIheight ; y++)
	{
		cv::Vec2d *rowPtr = locations[y];
		int *indicesPtr = indices[y];
		for (unsigned int x = 0; x < m_MSIwidth ; x++)
		{
			double *indPtr 	= modes[y * m_MSIwidth + x];

			rowPtr[x][0] = indPtr[0];
			rowPtr[x][1] = indPtr[1];
			indicesPtr[x] = m_maxModes[y * m_MSIwidth + x][0];
		}
	}
	
	double maxIntensity =0.0;
	double valx,valy =0.0;
	
	unsigned int ten = (m_MSIheight* m_MSIwidth)/10;
	int round = 1;
	std::cout << "  0 %" <<std::endl;	
	
	for(unsigned int y = 1; y < (m_MSIheight - 1U); y++)
	{
		double *x_ptr = distanceX[y];
		double *y_ptr = distanceY[y];
		cv::Vec2d *i_ptr = locations[y];
		cv::Vec2d *i_uptr = locations[y-1];
		cv::Vec2d *i_dptr = locations[y+1];
		double xx,yy;
		for(unsigned int x = 1; x < (m_MSIwidth - 1U); x++)
		{
			
			if(( (y*m_MSIwidth + x )% ten) == 0 )
			{	
				std::cout << " " << round * 10 << " %" <<std::endl;
				round++;
			}			
			
			{	// y-direction
				xx = (c1 * i_uptr[x-1][0] + c2 * i_uptr[x][0] + c3 * i_uptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_uptr[x][1] + c3 * i_uptr[x+1][1]) * fraction;
				cv::Point2d u(xx, yy);
				xx = (c1 * i_dptr[x-1][0] + c2 * i_dptr[x][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_dptr[x-1][1] + c2 * i_dptr[x][1] + c3 * i_dptr[x+1][1]) * fraction;
				cv::Point2d d(xx, yy);

				
				if(withUMap || withGraph)
				{
					unsigned int colU, rowU, colD, rowD, indexU, indexD;
					mapToNearestGridPoint(u, latentXCoords, latentYCoords, rowU, colU );
					mapToNearestGridPoint(d, latentXCoords, latentYCoords, rowD, colD );
					indexU = rowU * latentXCoords.cols + colU;
					indexD = rowD * latentXCoords.cols + colD;
					
					valy = graphDistance(indexU, indexD,withUMap);
				}	
				else	
					valy = vectorDistance(u, d);

				if (maxIntensity < valy)
					maxIntensity = valy;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valy = -valy;
				y_ptr[x] =  valy;
			}
			{	// x-direction
				xx = (c1 * i_uptr[x-1][0] + c2 * i_ptr[x-1][0] + c3 * i_dptr[x-1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_ptr[x-1][1] + c3 * i_dptr[x-1][1]) * fraction;
				cv::Point2d u(xx, yy);
				xx = (c1 * i_uptr[x+1][0] + c2 * i_ptr[x+1][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x+1][1] + c2 * i_ptr[x+1][1] + c3 * i_dptr[x+1][1]) * fraction;
				cv::Point2d d(xx, yy);
				
				if(withUMap || withGraph)
				{
					unsigned int colU, rowU, colD, rowD, indexU, indexD;
					mapToNearestGridPoint(u, latentXCoords, latentYCoords, rowU, colU );
					mapToNearestGridPoint(d, latentXCoords, latentYCoords, rowD, colD );
					indexU = rowU * latentXCoords.cols + colU;
					indexD = rowD * latentXCoords.cols + colD;
					
					valx = graphDistance(indexU, indexD,withUMap);
// 					valx = graphDistance(u, d);
				}	
				else	
					valx = vectorDistance(u, d);

				if (maxIntensity < valx)
					maxIntensity = valx;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valx = -valx;
				x_ptr[x] = valx;
			}
		}	
	
	}
	//normalization
	for (unsigned int y = 0; y < m_MSIheight ; y++)
  {
    double* x_ptr = distanceX[y];
    double* y_ptr = distanceY[y];
    for(unsigned int x = 0; x < m_MSIwidth ; x++)
    {
      x_ptr[x] =  ((x_ptr[x] + maxIntensity)*0.5/maxIntensity);
      y_ptr[x] =  ((y_ptr[x] + maxIntensity)*0.5/maxIntensity);
    }
  }
  std::cout << "100 %" <<std::endl;
}

void GTM::mapMagnifications( std::vector<cv::Mat> &latentMap, cv::Mat1d &magnifications )
{
	
	//dimension check
	assert(latentMap.size() == 2);
// 	assert(latentMap[0].rows == magnifications.rows && latentMap[0].cols == magnifications.cols);
// 	assert(latentMap[1].rows == magnifications.rows && latentMap[1].cols == magnifications.cols);
	cv::Mat magMap = cv::Mat::zeros(latentMap[0].rows,latentMap[0].cols, CV_64F );
	
	for(int y = 0; y < magMap.rows; y++)
	{

		//pointer to mapped magnifications 
		double *magMapPtr = magMap.ptr<double>(y);
		//pointer to latent x coordinate
		double *xLocPtr = latentMap[0].ptr<double>(y);
		//pointer to latent y coordinate
		double *yLocPtr = latentMap[1].ptr<double>(y);
		
		for(int x = 0; x < magMap.cols; x++)
		{
			//pointer to latent y position in magnifications
		
			double *magPtr = magnifications[(unsigned int)yLocPtr[x]];
			//pointer to latent x position in magnifications
			magMapPtr[x] = magPtr[(unsigned int)xLocPtr[x]];
		}
	}
	latentMap.push_back(magMap);
	
}

void GTM::umatrix(cv::Mat1d &magnifications)
{
	int width = magnifications.cols;
	int height = magnifications.rows;
		
	double *top;
	double *center;
	double *bottom;
		
	//touching border?
	bool l,t,r,b;

	int initialDegree;

	assert(m_conf->graph_type == "MESH");
	initialDegree = m_conf->sw_initialDegree;

	for(int y = 0; y < height; y++)
	{
		center 			= magnifications[y];
		top					= magnifications[y-1];
		bottom			= magnifications[y+1];
		for(int x = 0; x < width; x++)
		{
	
			t = false;
			l = false;
			r = false;
			b = false;
					
			if(y == 0 )
				t = true;
					
			if(x == 0)
				l = true;
				
			if(y == (height-1))
				b = true;
				
			if(x == (width-1))
				r = true;
				
			if(t == false)
			{	
				m_mesh->setEdgeWeight((y*width +x),((y-1)*width +x), std::abs(center[x] - top[x]));
			}
				
			if(l == false)
			{	
				m_mesh->setEdgeWeight((y*width +x),((y)*width +(x-1)), std::abs(center[x] - center[x-1]));

			}
				
			if(r == false)
			{	
				m_mesh->setEdgeWeight((y*width +x),((y)*width +(x+1)), std::abs(center[x+1] - center[x]));
			}
				
			if(b == false)
			{	
				m_mesh->setEdgeWeight((y*width +x),((y+1)*width +x), std::abs(center[x] - bottom[x]));
			}

//diagonal edges
			if((l | t) == false && (initialDegree == 8))
			{
				m_mesh->setEdgeWeight((y*width +x),((y-1)*width +(x-1)), std::abs(center[x] - top[x-1]) );
			}
			
			if((r | t) == false && (initialDegree == 8))
			{	
				m_mesh->setEdgeWeight((y*width +x),((y-1)*width +(x+1)), std::abs(center[x] - top[x+1]) );
			}
					
			if((b | l) == false && (initialDegree == 8))
			{	
				m_mesh->setEdgeWeight((y*width +x),((y+1)*width +(x-1)), std::abs(center[x] - bottom[x-1]));
			}
				
			if((b | r) == false && (initialDegree == 8))
			{	
				m_mesh->setEdgeWeight((y*width +x),((y+1)*width +(x+1)), std::abs(center[x] - bottom[x+1]));
			}	
		}
	}
		
	if(m_conf->withUMap)
		m_mesh->scaleDistances(m_conf->scaleUDistance);
	
}

std::string GTM::getFileExtension()
{
  std::stringstream s1, s2, s3, s4, s5, s6, s7, s8;
	//<name><height><width><samples><RBF><Latent><iterations><"GDIST"|"DDIST"><type>
	
  std::string height, width, samples, rbf, latent,iter, graphDistance;
  s1 << m_MSIheight;
  height = s1.str();
  s2 << m_MSIwidth;
  width = s2.str();
  s3 << m_conf->gtm_samplePercentage;
  samples = s3.str();	
  s4 << m_conf->gtm_numRbf;
  rbf = s4.str();
	s5 << m_conf->gtm_numLatent;
  latent = s5.str();
  s6 << m_conf->gtm_numIterations;
  iter = s6.str();
	//use magnification factors as edge weights
	if(m_conf->graph_withGraph)
	{	
		s7 << "_GDIST";
		s7 << "_N" << m_conf->sw_initialDegree;
		if(m_conf->withUMap)
		{	
			s7 << "_UMAP";
			s7 << "_" << m_conf->scaleUDistance;
		}	
	}	
	else
	// use direct distance between latent variables	
		s7 << "_DDIST";
	graphDistance = s7.str();	
	

  std::string name(m_conf->msi_name);
  
  if( name.find(".txt") ) name.resize(name.size() - 4);

  return name + "_"+height+"x"+width+"_S"+samples + "_RBF"+rbf+"_L"+latent+"_I"+iter + graphDistance;
}


GTM::~GTM()
{}