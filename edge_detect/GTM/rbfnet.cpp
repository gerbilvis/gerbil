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
#include "rbfnet.h"
#include "cv.h"



RBFNet::RBFNet(unsigned int nin, unsigned int nhidden, unsigned int nout, std::string func,double prior, S_RBFNet &net)
{
		std::cout << "nin " << nin << " nhidden " << nhidden << " nout " << nout << std::endl;
  assert(func == "GAUSSIAN" || func == "TPS" || func == "R4LOGR");
  
  net.type = "rbf";
  net.nin = nin;
  net.nhidden = nhidden;
  net.nout = nout;
  net.actfn = func;
	// Possible alternative : Neuroscale, for Sammon stress measure
  net.outfn = "LINEAR";
  m_prior = prior;
  
  net.nwts = nin * nhidden + ( (nhidden +1 ) * nout);
	 
	// Assume each function has a centre and a single width parameter, and that
	// hidden layer to output weights include a bias. Only the Gaussian function
	// requires a width
  if(func == "GAUSSIAN")
  {
		//Extra weights for width parameters
    net.nwts += nhidden;
  }  
  net.alpha = prior;
  
  //initialize parameter matrices
  net.c = cv::Mat::zeros(net.nhidden,net.nin, CV_64F);
  net.wi = cv::Mat::zeros(1,net.nhidden, CV_64F);
  net.w2 = cv::Mat::zeros(net.nhidden,net.nout, CV_64F);
  net.b2 = cv::Mat::zeros(1,net.nout, CV_64F);
   
  
//   initialize weight vector with zero mean and unit variance 
  cv::Mat1d w = cv::Mat::zeros(1, net.nwts, CV_64F);
//   cv::RNG rng(cv::getTickCount());
	uint64 state = 19;
	cv::RNG rng(state);

  rng.fill(w, cv::RNG::NORMAL, cv::Scalar(0.0),cv::Scalar(1.0));
  
//load random variables of matlab setup here!
// 	cv::Mat1d w;
// 	const char *fn_w  = "/home/knorx/UNI/Studienarbeit/working_directory/code/reflectance/ms_edges/GTM/data/rbf_w.txt";
// 	loadFromFile(w, fn_w);
	
  rbfunpak(net, w);
	
  if(func == "GAUSSIAN")
  {  
    net.wi.setTo(1.0);
  }  
  else
  {
    //TODO i.e. neuroscale, not implemented
  }
  
}

void RBFNet::rbfprior(std::string rbfunc, unsigned int nin, unsigned int nhidden, unsigned int nout, S_RBFNet &net)
{
  assert(net.actfn == "GAUSSIAN" || net.actfn == "TPS" || net.actfn == "R4LOGR");
  
  net.mask = cv::Mat::ones(net.nwts,1, CV_64F);
  
  unsigned int nwts_layer1 = 0U;
  
  if(rbfunc == "GAUSSIAN")
    nwts_layer1 = (nin * nhidden) + nhidden;
  
	else if(rbfunc == "TPS" || rbfunc == "R4LOGR")
		nwts_layer1 = (nin * nhidden);
	
	else
    std::cerr << "Undefined activation function" <<std::endl;
   
  for(unsigned int y = 0; y < net.nwts;y++ )
  {
    double *ptr = net.mask[0];
    if(y < nwts_layer1)
      ptr[y] = 0;
  }  
}

void RBFNet::rbfjacob(S_RBFNet &net, cv::Mat1d &latent_data, std::vector<cv::Mat1d> &jac)
{
	//function only implemented for linear outputs
	assert(net.outfn == "LINEAR");
	assert(net.actfn == "GAUSSIAN" || net.actfn == "TPS" || net.actfn == "R4LOGR");

	cv::Mat1d y, z, n2;
	cv::Mat1d psi;
	rbffwd(net, latent_data, y, z,n2); 
	unsigned int ndata = latent_data.rows;

	for(unsigned int i = 0; i < net.nout ; i++)
	{
		jac[i] = cv::Mat::zeros(ndata, net.nin, CV_64F);
	}	
	psi = cv::Mat::zeros(net.nin, net.nhidden, CV_64F);

	//calculate derivative w.r.t. n2
	cv::Mat1d dz = cv::Mat::zeros(ndata, net.nhidden, CV_64F);
	cv::Mat1d weights = cv::Mat::zeros(ndata, net.nhidden, CV_64F);
	cv::Mat1d weight_ones = cv::Mat::ones(ndata,1U, CV_64F);
	
	weights = (weight_ones *  net.wi);
	
	cv::Mat1d c_trans, x_trans, ones_nhidden, ones_nin;
	cv::transpose(net.c, c_trans);
	cv::transpose(latent_data, x_trans);
	ones_nhidden = cv::Mat::ones(1U,net.nhidden, CV_64F);
	ones_nin = cv::Mat::ones(net.nin ,1U, CV_64F);
	cv::Mat1d t1 = cv::Mat::zeros(net.nin, net.nhidden, CV_64F);
	cv::Mat1d t2 = cv::Mat::zeros(net.nin, net.nhidden, CV_64F);
	cv::Mat1d t3 = cv::Mat::zeros(net.nin, net.nout, CV_64F);
	
	if(net.actfn == "GAUSSIAN")
	{
		for(int y = 0; y < dz.rows; y++)
		{
			double *dzPtr = dz[y];
			double *zPtr 	= z[y];
			double *wPtr 	= weights[y];
			for(int x = 0; x < dz.cols; x++)
			{
				double val = (-1.0 * (zPtr[x]/ wPtr[x]));
				
				dzPtr[x] = val;
			}	
		}
	}
	else if(net.actfn == "TPS")
	{
		for(int y = 0; y < dz.rows; y++)
		{
			double *dzPtr = dz[y];
			double *n2Ptr = n2[y];

			for(int x = 0; x < dz.cols; x++)
			{
				double val = n2Ptr[x];
				if(val == 0.0)
					val = 1.0;
				
				dzPtr[x] = 2.0 * (1.0 + std::log(n2Ptr[x] + val));
			}	
		}	
	}	
	else if(net.actfn == "R4LOGR")
	{
		for(int y = 0; y < dz.rows; y++)
		{
			double *dzPtr = dz[y];
			double *n2Ptr = n2[y];

			for(int x = 0; x < dz.cols; x++)
			{
				double val = n2Ptr[x];
				if(val == 0.0)
					val = 1.0;
				
				dzPtr[x] = 2.0 * (n2Ptr[x] * (1.0 + 2.0 * std::log(n2Ptr[x] + val)));
			}	
		}			
	}
	else
		std::cerr << "Unknown activation function '" <<net.actfn << "'" <<std::endl;		
		
	//ignore biases as they cannot affect Jacobian
	for(unsigned int n = 0; n < ndata; n++)
	{
		t1 = ones_nin * (dz.row(n));
		t2 = x_trans.col(n) * ones_nhidden;
		t2 -= c_trans;
				
		cv::multiply(t1, t2, psi);
		t3 = psi * net.w2;
										 
		for(unsigned int dim = 0; dim < net.nout ; dim++)
		{
			double *planePtr = jac[dim].ptr<double>(n);
					
			for(unsigned int x = 0; x < net.nin ; x++)
			{
				planePtr[x] = t3[x][dim];
			}	
		}
	}
	
}


void RBFNet::rbfpak(S_RBFNet &net , cv::Mat1d &w)
{
  if(w.empty())
    w = cv::Mat::zeros(1, net.nwts, CV_64F);
  
  unsigned int mark1,mark2,mark3,mark4;
  
  //define borders for the parametes
  mark1 = net.nin * net.nhidden;
  mark2 = mark1 + net.nhidden;
  mark3 = mark2 + (net.nhidden * net.nout);
  mark4 = mark3 + net.nout;
  
  for(unsigned int i = 0; i < net.nwts; i++)
  {
    double *ptr = w[0];
    double *ptr_c = net.c[0];
    double *ptr_wi = net.wi[0];
    double *ptr_w2 = net.w2[0];
    double *ptr_b2 = net.b2[0];
    if(i < mark1)
      ptr[i] = ptr_c[i];
    else if(i >= mark1 && i < mark2 )
      ptr[i] = ptr_wi[i-mark1];
    else if(i >= mark2 && i < mark3 )
      ptr[i] = ptr_w2[i-mark2];
    else if(i >= mark3 && i < mark4 )
      ptr[i] = ptr_b2[i-mark3];   
  }  
  
}

void RBFNet::rbfunpak(S_RBFNet &net, cv::Mat1d w)
{
  assert((int)net.nwts == w.cols);
  
  unsigned int mark1,mark2,mark3,mark4;
  
  //define borders for the parametes
  mark1 = net.nin * net.nhidden;
  mark2 = mark1 + net.nhidden;
  mark3 = mark2 + (net.nhidden * net.nout);
  mark4 = mark3 + net.nout;
  
  for( unsigned int i = 0; i < net.nwts; i++)
  {

    double *ptr = w[0];
    double *ptr_c = net.c[0];
    double *ptr_wi = net.wi[0];
    double *ptr_w2 = net.w2[0];
    double *ptr_b2 = net.b2[0];
    
    if(i < mark1)
      ptr_c[i] = ptr[i];
    else if(i >= mark1 && i < mark2 )
      ptr_wi[i-mark1] = ptr[i];
    else if(i >= mark2 && i < mark3 )
      ptr_w2[i-mark2] = ptr[i];
    else if(i >= mark3  )
      ptr_b2[i-mark3] = ptr[i];    
    
  }
  
	
	//correct order of matrices net.c and net.w2
	for( unsigned int x = 0; x < net.nin; x++)
	{
		for( unsigned int y = 0; y < net.nhidden; y++)
		{
			net.c[y][x] = w[0][(x * net.nhidden + y)];
		}
	}	
	
	for( unsigned int x = 0; x < net.nout; x++)
	{
		for( unsigned int y = 0; y < net.nhidden; y++)
		{
			net.w2[y][x] = w[0][(x * net.nhidden + y+ mark2)];
		}
	}	
  
}

void RBFNet::dumpNet(S_RBFNet &net)
{
  //dump parameters
  std::cout << "+++ Content of NET structure +++\n" <<std::endl;
  std::cout << "Type: " <<net.type <<std::endl;
  std::cout << "nin: " <<net.nin <<std::endl;
  std::cout << "nhidden: " <<net.nhidden <<std::endl;
  std::cout << "nout: " <<net.nout <<std::endl;
  std::cout << "nwts: " <<net.nwts <<std::endl;
  std::cout << "alpha: " <<net.alpha <<std::endl;
  std::cout << "actfn: " <<net.actfn <<std::endl;
  
  
  std::cout << "\nc matrix: " <<net.nhidden << " x " <<net.nin <<std::endl<<std::endl;
  std::cout << "\nc matrix: " <<net.c <<std::endl;
  
  std::cout << "\nwi matrix: " <<  "1 x " << net.nhidden <<std::endl<<std::endl;
	std::cout << "\nwi matrix: " <<net.wi <<std::endl;
  
  std::cout << "\nw2 matrix: " <<net.nhidden << " x " << net.nout  <<std::endl<<std::endl;
  std::cout << "\nw2 matrix: " <<net.w2 <<std::endl;
  
  std::cout << "\nb2 matrix: " << "1 x " << net.nout <<std::endl<<std::endl;
  std::cout << "\nb2 matrix: " <<net.b2 <<std::endl;
  
  std::cout << "\nmask: " << net.nwts << " x 1"  <<std::endl<<std::endl;
  std::cout << "\nmask matrix: " <<net.mask <<std::endl; 

  
}

RBFNet::~RBFNet()
{}
    
void RBFNet::rbfsetfw(S_RBFNet &net, double scale )
{
  
  //set the  variances to the largest squared distances between centres
  if(net.actfn == "GAUSSIAN")
  {
    cv::Mat1d cdist;
    double min_per_col = 0.0f;
    double max = 0.0f;
    double min = DBL_MAX;
    double widths = 0.0f;
    dist2(net.c, net.c, cdist);
				
    if(scale > 0.0f)
    {
      //set variance of the basis  to be a scale times average distance to nearest neighbor
      for(int x = 0; x < cdist.cols; x++)
      {
        //set diagonal to largest value
        for(int y = 0; y < cdist.rows; y++)
        {
          if(y == x)
            cdist[y][x] = DBL_MAX;
          
          //calcualte minimum per column
          if(cdist[y][x] < min)
          {
            min = cdist[y][x];
          }
          
        }
        min_per_col += min;
      }
      min_per_col/=net.nhidden;
      widths = scale * min_per_col;
    
    }
    else
    {
      cv::minMaxLoc(cdist, NULL, &max);
      widths = max;
    }

    net.wi.setTo(widths);
  } 
}


//outputs = A phi = Phi distances = n2
void RBFNet::rbffwd(S_RBFNet &net, cv::Mat1d &x, cv::Mat1d &a, cv::Mat1d &z, cv::Mat1d &n2 )
{
	assert(net.actfn == "GAUSSIAN" || net.actfn == "TPS" || net.actfn == "R4LOGR");
	
	unsigned int ndata = x.rows;

	dist2(x, net.c, n2);
	
	cv::Mat1d wi2;
	z = cv::Mat::zeros(x.rows, net.wi.cols,CV_64F);
	
	if(net.actfn == "GAUSSIAN")
	{
		//calculate widths factors
		wi2 = cv::Mat::zeros(x.rows, net.wi.cols,CV_64F);
				
		for(int y = 0; y < x.rows; y++)
		{
			double *rowptr = wi2[y];
			for(int x = 0; x < net.wi.cols; x++)
			{
				double *wi_ptr = net.wi[0];
				rowptr[x] = (2.0 * wi_ptr[x]);
			}
		}	

		//compute activations
		for(int y = 0; y < x.rows; y++)
		{
			double *zPtr = z[y];
			double *wi_ptr = wi2[y];
			double *dist_ptr = n2[y];
			for(int x = 0; x < net.wi.cols; x++)
			{
				
				zPtr[x] = std::exp(- (dist_ptr[x] / wi_ptr[x]));
			}
		}
		
	}	
	else if(net.actfn == "TPS")
	{
		
		for(int y = 0; y < z.rows; y++)
		{
			double *zPtr = z[y];
			double *n2Ptr = n2[y];
			for(int x = 0; x < z.cols; x++)
			{
				double val = zPtr[x];
				if(val == 0.0)
					val = 1.0;
				
				zPtr[x] = n2Ptr[x] * std::log(n2Ptr[x] + val );
			}	
		}
		
	}
	else if(net.actfn == "R4LOGR")
	{
		for(int y = 0; y < z.rows; y++)
		{
			double *zPtr = z[y];
			double *n2Ptr = n2[y];
			for(int x = 0; x < z.cols; x++)
			{
				double val = zPtr[x];
				if(val == 0.0)
					val = 1.0;
				
				zPtr[x] = n2Ptr[x] * n2Ptr[x] *std::log(n2Ptr[x] + val );
			}	
		}		
	}	
	else
		std::cerr << "Unknown activation function in RBFNet::rbffwd() " <<std::endl;
	
	a = cv::Mat::zeros(ndata, net.w2.cols,CV_64F);
	cv::Mat1d ndata_ones = cv::Mat::ones(ndata,1,CV_64F); 
	
	//summation of outputs
	a = z * net.w2;
	//add bias
	a += (ndata_ones * net.b2);

}
