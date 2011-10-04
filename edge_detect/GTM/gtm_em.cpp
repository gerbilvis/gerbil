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



#include "gtm_em.h"

#include <iostream>
 
EM::EM(const EdgeDetectionConfig *config, S_GTMNet &net, cv::Mat1d &t, FOptions &options, bool store)
{
	m_config = config;
	m_store = store;
	m_test = false;
	
	if(options.max_number_function_eval != 0U)
		m_niters = options.max_number_function_eval;
	else
		m_niters = 100U;
	
	switch(options.display_parameter)
	{
		case -1:m_verbosity = OFF; break;
		case 0:	m_verbosity = ON; break;
		case 1:	m_verbosity = HIGH; break;

		default: m_verbosity = OFF; break;
	}	
	
	//if store == true, error values are stored here
	errlog = cv::Mat::zeros(1,(m_niters+1),CV_64F);

	//if test == true the log-likelihood is tested for termination
	if(options.termination_tolerance_F > 0.0)
		m_test = true;
	
	//Calculate various quantities that remain constant during training
	unsigned int ndata 	= t.rows;
	unsigned int tdim		=	t.cols;
	
	unsigned int ND = ndata * tdim;
	
	cv::Mat1d phi,phi_ext,phi_ext_trans, distances;

	net.rbf->rbffwd(net.rbfnet, net.X, net.gmmnet.centres, phi, distances);
	phi_ext = cv::Mat::ones(phi.rows, (phi.cols+1U),CV_64F);
	
	for(int y = 0; y < phi_ext.rows;y++)
	{
		double *rowptr = phi_ext[y];
		double *phiptr = phi[y];
		for(int x = 0; x < phi.cols;x++)
		{
			rowptr[x] = phiptr[x];
		}	
	}	
	
	cv::transpose(phi_ext, phi_ext_trans);
	
	cv::Mat1d A = cv::Mat::zeros(phi_ext.rows, phi_ext.cols,CV_64F);
	cv::Mat1d Alpha = cv::Mat::zeros(phi_ext.cols, phi_ext.cols,CV_64F);
	cv::Mat1d W = cv::Mat::zeros(phi_ext.cols, t.cols,CV_64F);
	
	
	//use a sparse representation for the weight regularization matrix
	if(net.rbfnet.alpha > 0.0)
	{
		Alpha = (net.rbfnet.alpha * cv::Mat::eye(phi_ext.cols, phi_ext.cols,CV_64F));
		Alpha[(phi_ext.cols-1)][(phi_ext.cols-1)] = 0.0;
	}	
	
	//do the actual EM
	cv::Mat1d priors_trans;
	cv::Mat1d prob;
	cv::Mat1d max;
	cv::Mat1d log_max;
	cv::Scalar sum_log_max;
	double e = DBL_MAX;//error value
	double e_old = DBL_MAX;
	infoStream << "#Beta " << std::endl;
	
	for(unsigned int i = 0; i < m_niters;i++)
	{
		//calculate responsibilities
		cv::Mat1d R, act, postN;
		net.gtm->gtmpost(net, t, R, act);
		std::string fn = m_config->output_dir + net.gtm->getFileExtension() ;
// 		writePosterior(R,cv::Size(net.gtm->latentWidth(),net.gtm->latentHeight()), 250 ,postN ,i,fn );
		
		//calculate error value if needed
		if( m_verbosity == ON || m_verbosity == HIGH || m_store == true || m_test == true)
		{
			
			cv::transpose(net.gmmnet.priors, priors_trans);
			prob = act * priors_trans;
			
			//Error value is negative log-likelihood of data
			max = maxVS(prob, DBL_EPSILON);
			
			cv::log(max, log_max);
			sum_log_max = cv::sum(log_max);

			e = - sum_log_max[0];
			
			if(m_store)
				errlog[0][i] = e;
			
			if(m_verbosity == HIGH)
				std::cout << "Cycle "<<i << "\tError: " << e <<std::endl;
			
			if(m_test)
			{
				if( (i > 0) & (std::abs(e- e_old ) < options.termination_tolerance_F ))
				{	
					options.function_value_lambda = e;
					break;
				}
				else
				{
					e_old = e;
				}	
			}	
		}
		//Calculate matrix be inverted (phi' * G * phi + alpha * I in the papers).
		//Sparse representation of G normally executes faster and saves memory
		cv::Mat1d sum_col;
		cv::Mat1d diag_sum_col;
		cv::Mat1d sum_col_trans;
		
		sumCol(R,sum_col);
		cv::transpose(sum_col, sum_col_trans);
		
		if(net.rbfnet.alpha > 0.0)
		{
// 			A = phi_ext_trans * diagV(sum_col) * phi_ext;
			cv::Mat1d tmp = sparseMul(phi_ext_trans, sum_col_trans);

			A = (tmp * phi_ext);
			A += (net.gmmnet.covars[0][0] * Alpha);
		
		}
		else
		{
			cv::Mat1d tmp = sparseMul(phi_ext_trans, sum_col_trans);
			A = (tmp * phi_ext);
// 			A = phi_ext_trans * diagV(sum_col) * phi_ext;		
		}
		//A is a symmetric matrix likely to be positiv definite, so try
		//fast Cholesky decomposition to calculate W, otherwise use SVD.
		// (phi' * (R* t)) is computed right-to-left, as R and t are 
		//normally (much) larger than phi'
		cv::Mat1d cholDcmp, svdDcmp;
		double singular;
		cv::Mat1d R_trans;
		cv::Mat1d t1,t2;
		
		t1 = cv::Mat::zeros(R_trans.rows, t.cols, CV_64F);
		cv::transpose(R, R_trans); 
		
		t1 = R_trans * t;
	
		t2 = phi_ext_trans * t1;
		
		//try cholesky decomposition
		singular = cv::invert(A, cholDcmp, cv::DECOMP_CHOLESKY);
		if(0.0 == singular )
		{	
			if(m_verbosity == ON || m_verbosity == HIGH)
			{
				std::cerr << "gtm_em: Warning -- M-Step Matrix singular, using SVD" <<std::endl;
			}
			
			//compute pseudo-inverse
			cv::invert(A, svdDcmp, cv::DECOMP_SVD);

			W = svdDcmp * t2;
		}
		else
		{
			W = cholDcmp * t2;
		}	
		
		//Put the new weights into the network to calculate responsibilities
		unsigned int nhidden = net.rbfnet.nhidden;
		unsigned int wcols = W.cols;
		unsigned int wrows = W.rows;
		for(unsigned int y = 0; y < wrows; y++)
		{
			double *rowPtr = W[y];
			double *weightPtr = net.rbfnet.w2[y];
			double *biasPtr = net.rbfnet.b2[0];
			for(unsigned int x = 0; x < wcols; x++)
			{
				if( y < nhidden)
				{
					weightPtr[x] = rowPtr[x];
				}
				else
					biasPtr[x] = rowPtr[x];
			}	
		}
		
		//calculate new distances
		cv::Mat1d distances, t3, mul_distR;
		dist2(t, (phi_ext * W), distances);
		
		//calculate new value for beta
		cv::Scalar sum_distR;
		cv::multiply(distances, R, mul_distR);
		
		sum_distR = cv::sum(mul_distR);
		
		//normalize by number of inputs * input dimension
		sum_distR[0] /= ND;
		infoStream << sum_distR[0] <<std::endl;
		net.gmmnet.covars = (cv::Mat::ones(1U, net.gmmnet.ncentres,CV_64F) * sum_distR[0]);
	}	
	
	cv::Mat1d gmmprob, log_gmm_prob;
	cv::Scalar sum_log_gmm_prob;
	net.gtm->gtmprob(net, t, gmmprob);
	
	cv::log(gmmprob, log_gmm_prob);
	sum_log_gmm_prob = cv::sum(log_gmm_prob);
	
	
	options.function_value_lambda = -sum_log_gmm_prob[0];
	errlog[0][m_niters] = -sum_log_gmm_prob[0];
	
	if(m_verbosity == HIGH)
	{	
		std::cout << "Maximum number of iterations has been exceeded" <<std::endl;
		std::cout << "Function value "<< options.function_value_lambda <<std::endl;
	}
	
	if(m_store)
	{
		std::string s_err 	= m_config->output_dir +  net.gtm->getFileExtension() + "_errlog.dat";
		const char* fn_err = s_err.c_str();
		infoStream << std::endl <<  writeError(errlog, fn_err,std::ofstream::trunc);
	}	
}

EM::~EM(){};
