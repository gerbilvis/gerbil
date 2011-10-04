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



#include <cassert> 

#include "gtm_misc.h" 


void drawSamples(cv::Mat1d &src, double samplePercentage, cv::Mat1d &dest,cv::Size imgDimension, bool fixedSeed)
{

	assert(samplePercentage >= 0.0 && samplePercentage <= 1.0);
	
	unsigned int width = imgDimension.width;
	unsigned int height = imgDimension.height;
	unsigned int nsamples = std::ceil(samplePercentage * (double)(width * height));
	dest = cv::Mat::zeros(nsamples, src.cols,CV_64F);
	
	cv::Mat1i seed = cv::Mat::zeros(1, imgDimension.width* imgDimension.height,CV_32S);
	for(int n = 0; n < seed.cols; n ++)
	{
		seed[0][n] = n;
	}	

	unsigned int sampleDim = src.cols;
		
	cv::RNG rng;	
	uint64 state = 19;
	if(fixedSeed)
		rng = cv::RNG(state);
	else
		rng = cv::RNG(cv::getTickCount());
		// generate random sequence of the input x,y range
// 		rng.fill( shuffledY, cv::RNG::UNIFORM, cv::Scalar(0), cv::Scalar(width*height));
	cv::randShuffle(seed, 5.0, &rng);

	for(unsigned int n = 0; n < nsamples;n++)
	{
		int rand = seed[0][n];
		double *destPtr = dest[n];
		double *srcPtr = src[rand];
		for(unsigned int dim = 0; dim < sampleDim; dim++)
		{
			destPtr[dim] = srcPtr[dim];
		}	
	}

}

void mapToNearestGridPoint(cv::Point2d &point, cv::Mat1d &latentX,cv::Mat1d &latentY, unsigned int &row, unsigned int &col)
{
	
	//! function to map discrete values ('point')to intervalls/nearest grid point in the latent grid
	assert( latentX.rows == latentY.rows && latentX.cols == latentY.cols);
	
	double delta = 2.0 / (double(latentX.cols)-1.0);
	double nx = (1.0 + point.x)/delta;
	double ny = (1.0 - point.y)/delta;
	double intpart = 0.0;
	
  if(std::modf(nx,&intpart) >=0.5)
    col = (unsigned int)std::ceil(nx);
  else
    col = (unsigned int)std::floor(nx);
	
  if(std::modf(ny,&intpart) >=0.5)
    row = (unsigned int)std::ceil(ny);
  else
    row = (unsigned int)std::floor(ny);	
	
	//may be caused due to numerical insufficiencies of modf() !!
	if(ny < 0.0)
		row = 0U;

}

void msi2Mat(multi_img &img, cv::Mat1d &dest)
{
	unsigned int ndata = img.height * img.width;
	unsigned int width = img.width;
	unsigned int height = img.height;
	unsigned int ndim = img.size();
	dest = cv::Mat::zeros(ndata, img.size(), CV_64F);

	for(unsigned int y = 0; y < height; y++)
	{	
		for(unsigned int x = 0; x < width; x++)
		{		
			const multi_img::Pixel &p = img(y,x);
			for(unsigned int dim = 0; dim < ndim; dim++)
			{
				dest[(y * width +x) ][dim] = (double)p[dim];
			}	
		}	
	}	
}

double vectorDistance(const cv::Point2d &p1, const cv::Point2d &p2)
{
	double dx, dy;
	
	dx = (p1.x - p2.x); dy = (p1.y - p2.y);
	return std::sqrt(dx * dx + dy * dy);
}

double vectorDistance3d(const cv::Point3d &p1, const cv::Point3d &p2)
{
	double dx, dy, dz;
	
	dx = (p1.x - p2.x); 
	dy = (p1.y - p2.y);
	dz = (p1.z - p2.z);
	return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void dist2(cv::Mat1d x,cv::Mat1d c, cv::Mat1d &dest)
{

  unsigned int ndata,dimx,ncentres,dimc;
  
  ndata = x.rows;
  dimx  = x.cols;
  ncentres = c.rows;
  dimc = c.cols;
  
  assert(dimx == dimc);

  cv::Mat1d sum1 = cv::Mat::zeros(ndata,1U, CV_64F);

  cv::Mat1d sum2 = cv::Mat::zeros(ncentres,1U, CV_64F);

	cv::Mat1d sum1_ones = cv::Mat::ones(ncentres,1,CV_64F);

	cv::Mat1d sum2_ones = cv::Mat::ones(ndata,1,CV_64F);

	cv::Mat1d tmp = cv::Mat::zeros(ndata,ncentres, CV_64F);

	dest = cv::Mat::zeros(ndata,ncentres, CV_64F);
	
	cv::Mat1d c_trans,x_trans;
	cv::transpose(x, x_trans);
	cv::transpose(c, c_trans);
	  
  for(unsigned int ym = 0; ym < ndata; ym++)
  {
    for(unsigned int xm = 0; xm < dimx; xm++)
    {
      sum1[ym][0] += (x[ym][xm] * x[ym][xm]);
    }
  }

	cv::transpose(sum1, x_trans);
	
	tmp = sum1_ones * x_trans;
	cv::transpose(tmp, dest);

  for(unsigned int ym = 0; ym < ncentres; ym++)
  {
    for(unsigned int xm = 0; xm < dimc; xm++)
    {
      sum2[ym][0] += (c[ym][xm] * c[ym][xm]);
    }
  }

  cv::transpose(sum2, c_trans);

	dest += (sum2_ones * c_trans);

	cv::transpose(c, c_trans);

	dest -=  (x * c_trans);
	dest -=  (x * c_trans);
	
	for(int ym = 0; ym < dest.rows; ym++)
  {
		double* rowptr = dest[ym];
    for(int xm = 0; xm < dest.cols; xm++)
    {
			if(rowptr[xm] < 0.0)
				rowptr[xm] = 0.0;
    } 
	} 

}


void sumCol(cv::Mat1d &src, cv::Mat1d &dest)
{
	dest = cv::Mat::zeros(1, src.cols, CV_64F);

// 	
	for(int x = 0; x < src.cols; x++)
	{	
		double val = 0.0;
		for(int y = 0; y < src.rows; y++)
		{
			if(src[y][x] == NAN)continue;
			val += src[y][x]; 
		}
		dest[0][x]	= val;
	}

}


void sumRow(cv::Mat1d &src, cv::Mat1d &dest)
{
	dest = cv::Mat::zeros(src.rows, 1, CV_64F);
	
	for(int y = 0; y < src.rows; y++)
	{	
		double *rowPtr = src[y];
		double *destPtr = dest[y];
		
		for(int x = 0; x < src.cols; x++)
		{
			destPtr[x] += rowPtr[x];
		}
	}

}


cv::Mat1d diagV(cv::Mat1d &src)
{
	//TODO we first use row vectors as input, an additional input flag could decide if src is row or column vector
	assert(src.cols > 1);
	cv::Mat1d dest = cv::Mat::zeros(src.cols, src.cols, CV_64F);
	
	for(int y = 0; y < src.cols ; y++)
	{	
		double *rowPtr = dest[y];
		double *srcPtr = src[0];
		for(int x = 0; x < src.cols ; x++)
		{
			if(y == x)
				rowPtr[x] = srcPtr[x];
		}	
	}
	return dest;
}

cv::Mat1d sparseMul(cv::Mat1d &src1, cv::Mat1d &src2)
{
	assert(src2.cols == 1);
	
	cv::Mat1d res = cv::Mat::zeros(src1.rows, src2.rows, CV_64F);
	
	for(int y = 0; y < res.rows; y++)
	{
		double *src1Ptr = src1[y];
		double *resPtr = res[y];		
		
		for(int x = 0; x < res.cols; x++)
		{
			resPtr[x] = src1Ptr[x] * src2[0][x];
		}	
	}
	
	return res;
}

cv::Mat1d maxVS(cv::Mat1d &src1, double src2)
{
	//we want only compare vector with scalar here
	assert(src1.cols ==1);
	cv::Mat1d res = cv::Mat::zeros(src1.rows, src1.cols, CV_64F);
	
	
	for(int y = 0; y < src1.rows; y++)
	{
		res[y][0] = std::max(src1[y][0] , src2);
	}
	
	return res;
}

void scaleEVec(cv::Mat1d &eigenvectors, cv::Mat1d &eigenvalues,cv::Mat1d &dest, unsigned int dim_latent)
{
	cv::Mat1d lowDimEvalues = cv::Mat::zeros(eigenvectors.rows, dim_latent, CV_64F);
	cv::Mat1d diag = cv::Mat::zeros(dim_latent, dim_latent, CV_64F);
	dest = cv::Mat::zeros(eigenvectors.rows, dim_latent, CV_64F);

	//build diagonal matrix, with dim_laten largest eigenvalues and calculate square root
	for(int y = 0; y < diag.rows; y++)
	{	
		double *rowptr = diag[y];
		double *srcptr = eigenvalues[y];
		for(int x = 0; x < diag.cols; x++)
		{
			if(y == x)
				rowptr[x] = std::sqrt(srcptr[x]);
		}
	}	
	
	for(int y = 0; y < lowDimEvalues.rows; y++)
	{
		double *rowptr_ev = eigenvectors[y];
		double *rowptr = lowDimEvalues[y];
		for(int x = 0; x < lowDimEvalues.cols; x++)
		{
			rowptr[x] = (rowptr_ev[x]);
		}	
	}
		
  for(int ym = 0; ym < dest.rows; ym++)
  {
    for(int xm = 0; xm < dest.cols; xm++)
    {
			
      for(unsigned int i = 0; i < dim_latent; i++)
      { 
        dest[ym][xm] += (lowDimEvalues[ym][i] * diag[i][xm]);
      } 
    } 
	} 
}

void diag(cv::Mat1d &dest , unsigned int dimension, double value )
{
	dest = cv::Mat::zeros(dimension, dimension, CV_64F);
	for(unsigned int y = 0; y < dimension; y++)
	{
		double *rowptr = dest[y];
		for(unsigned int x = 0; x < dimension; x++)
		{
			if(y == x )rowptr[x] = value;
		}	
	}	
}

void normalize(cv::Mat1d &X, cv::Mat1d &normX)
{
	cv::Mat1d X_ones = cv::Mat::ones(X.rows, X.cols, CV_64F);
	normX = cv::Mat::ones(X.rows, X.cols, CV_64F);
	cv::Mat1d diag_mean,diag_std;
	cv::Scalar mean, stdv;
	
	cv::meanStdDev(X, mean, stdv);
	
	diag(diag_mean, X.cols, mean[0]);
	diag(diag_std, X.cols, 1.0/stdv[0]);
	
	X_ones *= diag_mean;
	
	X.copyTo(normX);
	
	//remove mean
	normX -=X_ones;
	
	//scale by std deviation
	normX *= diag_std;
}

void loadFromFile(cv::Mat1d &dest, const char *fn)
{
	std::ifstream in;
	std::string s;
	//stores lines to parse
	std::vector<std::string> data_row;
	
	bool firstline = true;
	unsigned int rows = 0;
	unsigned int cols = 0;

	in.open(fn, std::ifstream::in);

	assert(in.is_open());

	size_t found;
	// find out data dimension
	while(!in.eof())
	{
		getline(in, s);
		if(s !="")
		{
			if(firstline)
			{
				found = s.find('\t');
				cols = 1;
				while(found != std::string::npos)
				{
					found = s.find('\t', found+1);
					if(found !=std::string::npos)cols++;
				}	
				
				firstline=false;
			}	
			rows++;
			data_row.push_back(s);
		}	
		
	}
	
	dest = cv::Mat::zeros(rows, cols ,CV_64F);
	for(unsigned int y = 0; y < data_row.size() ;y++)
	{
		std::string toParse =  data_row.at(y);
		double *rowptr = dest[y];
		std::vector<std::string> data_col = tokenize(toParse, "\t");

		assert((data_col.size()-1) == cols);
		for(unsigned int x = 0; x < cols ;x++)
		{
			std::stringstream ss;
			ss << data_col.at(x);

			double val;
			ss >> val;
			rowptr[(x)] = val;
			
		}	
	}	
}

std::vector<std::string> tokenize(const std::string& str,const std::string& delimiters)
{
	std::string client = str;
	std::vector<std::string> result;

	while (!client.empty())
	{
		std::string::size_type dPos = client.find_first_of( delimiters );
		if ( dPos == 0 ) 
		{ // head is delimiter
			client = client.substr(delimiters.length()); // remove header delimiter
			result.push_back("");
		} 
		else 
		{ // head is a real node
			std::string::size_type dPos = client.find_first_of( delimiters );
			std::string element = client.substr(0, dPos);
			result.push_back(element);

			if (dPos == std::string::npos) 
			{ // node is last element, no more delimiter
				return result;
			} 
			else 
			{
				client = client.substr(dPos+delimiters.length());
			}
		}
	}
	
	if (client.empty()) 
	{ // last element is delimeter
		result.push_back("");
	}
	return result;
}

void maxVal(cv::Mat1d &src, double &val, unsigned int &pos )
{
	assert(src.cols == 1);
	val =0.0;
	pos = UINT_MAX;
	for(int y = 0; y < src.rows; y++)
	{
		double *rowptr = src[y];
		
		if(rowptr[0] > val)
		{
			val = rowptr[0];
			pos = y;
		}
	}	
}

void reshapeVM(cv::Mat1d &src, cv::Mat1d &dest, int rows, int cols, bool rowInput)
{
	dest = cv::Mat::zeros(rows, cols, CV_64F);
	
	if(rowInput)
	{	
		assert(( (src.cols * src.rows) == (rows * cols)) && (src.rows == 1));	
		for(int y = 0; y < rows;y++)
		{
			double *srcPtr = src[0];
			double *destPtr = dest[y];
			for(int x = 0; x < cols;x++)
			{
				destPtr[x] = srcPtr[(x * rows + y)];
			}	
		}	
	}
	else
	{
		assert(( (src.cols * src.rows) == (rows * cols)) && (src.cols == 1));
		
		for(int x = 0; x < cols;x++)
		{
					
			for(int y = 0; y < rows;y++)
			{
				dest[y][x] = src[x * rows + y][0];
			}
		}	
	}	
}

void squeeze(std::vector<cv::Mat1d> &ndim, cv::Mat1d &oneDim, unsigned int nout, unsigned int row)
{
	assert(ndim.size() > 1);
	unsigned int ndimCols = ndim[0].cols;
	oneDim = cv::Mat::zeros(ndimCols, nout, CV_64F);

	for(unsigned int x = 0; x < ndimCols ; x++)
	{
				
		for(unsigned int dim = 0; dim < nout ; dim++)	
		{
			double *planePtr = ndim[dim].ptr<double>(row);
			oneDim[x][dim] = 	planePtr[x];
		}	
	}
}

void writeMeans(cv::Mat1d &means, cv::Mat1d &labels, const char* fn,std::ios_base::openmode mode)
{
	
	std::ofstream out;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "# Means of data "<< std::endl;
		out << "# Means(1)\tMeans(2)\tClass"<< std::endl;
	}
	
	for(int y = 0; y < means.rows; y++ )
	{
		double *meansPtr = means[y];
		double *labelsPtr = labels[y];
		out << meansPtr[0]	 <<"\t" << meansPtr[1]<< "\t" << labelsPtr[0] <<std::endl;
	}
	out.close();
}

void writeModes(cv::Mat1d &modes,cv::Mat1i &indexMaxR, cv::Mat1d &labels, const char* fn,std::ios_base::openmode mode)
{
	std::ofstream out;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "# Modes of data "<< std::endl;
		out << "# Modes(1)\tModes(2)\tClass"<< std::endl;
	}	
	
	for(int y = 0; y < modes.rows; y++ )
	{
		double *modesPtr = modes[y];
		int *indPtr = indexMaxR[y];
		out << modesPtr[0]	 <<"\t" << modesPtr[1]<< "\t" << indPtr[0] <<std::endl;
	}
	out.close();
}

void writeMeansModes(cv::Mat1d &means,cv::Mat1d &modes, cv::Mat1d &labels, const char* fn,std::ios_base::openmode mode)
{
	std::ofstream out;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "# Combined Means and Modes of data "<< std::endl;
		out << "# Means(1)\tModes(1)\tMeans(2)\tModes(2)\tClass"<< std::endl;
	}	
	
	for(int y = 0; y < modes.rows; y++ )
	{
		double *modesPtr = modes[y];
		double *meansPtr = means[y];
		double *labelsPtr = labels[y];
		out << meansPtr[0] <<"\t" << meansPtr[1]<<"\t"	<< labelsPtr[0] <<"\n" << modesPtr[0]<<"\t" << modesPtr[1]<< "\t" <<labelsPtr[0] << "\n\n"  <<std::endl;
	}
	out.close();
}

void writePosterior(cv::Mat1d &X, cv::Mat1d &Y,cv::Mat1d post, const char *fn, std::ios_base::openmode mode)
{
	std::ofstream out;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "# Posteriors of data "<< std::endl;
		out << "# Latent x\tLatent y\tResponsibility\tClass\t"<< std::endl;
	}	
	
	for(int y = 0; y < post.rows; y++ )
	{
		
		double *yPtr = Y[y];
		double *xPtr = X[y];
		double *postPtr = post[y];
				
		for(int x = 0; x < post.cols; x++ )
		{
			out << xPtr[x] <<"\t" << yPtr[x] <<  "\t" <<postPtr[x]<<"\t" << std::endl;
		}
		out << "\n";

	}
	out.close();
}

void writePosterior(cv::Mat1d &Responsibilities,cv::Size latentGrid,  unsigned int n,cv::Mat1d &postN ,unsigned int iter, std::string fn)
{
// 	std::ofstream out;
	std::stringstream iteration;
	iteration << "_" << iter;
	fn += iteration.str();
	fn += ".png";
// 	out.open(fn, std::ofstream::out | mode);
// 	assert(out.is_open());

// 	if(mode == std::ofstream::trunc)
// 	{	
// 		out << "# Posteriors of data "<< std::endl;
// 		out << "# Latent x\tLatent y\tResponsibility\"<< std::endl;
// 	}
	
	cv::Mat1d rowN = Responsibilities.row(n); 
	
	reshapeVM(rowN, postN, latentGrid.height, latentGrid.width , true);
	
	cv::Mat_<uchar> showPost(postN.rows,postN.cols);
	double max = 0.0;
	for(int y = 0; y < postN.rows; y++ )
	{
		double *postPtr = postN[y];
			
		for(int x = 0; x < postN.cols; x++ )
		{
			if(postPtr[x] > max)max = postPtr[x];
		}
	}	
	
	
	for(int y = 0; y < postN.rows; y++ )
	{
		double *postPtr = postN[y];
		uchar *showPostPtr = showPost.ptr<uchar>(y);
		
				
		for(int x = 0; x < postN.cols; x++ )
		{
			showPostPtr[x] = static_cast<uchar>(255.0 *  postPtr[x]/max);
		}
		cv::imwrite(fn,showPost );
	}
// 	out.close();	
}

void writeMagnifications(cv::Mat1d &X, cv::Mat1d &Y,cv::Mat1d &mag, const char *fn, std::ios_base::openmode mode)
{
	std::ofstream out;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "# Magnifications of Latent variables "<< std::endl;
		out << "# Latent x\tLatent y\tMagnification\t"<< std::endl;
	}	
		
	for(int y = 0; y < mag.rows; y++ )
	{
		
		double *yPtr = Y[y];
		double *xPtr = X[y];
		double *magPtr = mag[y];
				
		for(int x = 0; x < mag.cols; x++ )
		{
			out << xPtr[x] <<"\t" << yPtr[x] <<  "\t" <<magPtr[x]<<"\t" << std::endl;
		}
		out << "\n";

	}
	out.close();
}

std::string writeError(cv::Mat1d &errlog, const char *fn, std::ios_base::openmode mode)
{
	std::ofstream out;
	std::stringstream res;
	
	out.open(fn, std::ofstream::out | mode);
	assert(out.is_open());

	if(mode == std::ofstream::trunc)
	{	
		out << "#Errorlog "<< std::endl;
		res <<"#Errorlog "<< std::endl;
		out << "#Iteration \tError\t"<< std::endl;
	}	
	
	double *errPtr = errlog[0];
	for(int x = 0; x < (errlog.cols-1); x++ )
	{
			out << errPtr[x] << std::endl;
			res << errPtr[x] << std::endl;
	}
	out << "Funtion value" <<std::endl;
	res << "Funtion value" <<std::endl;
	out <<  errPtr[errlog.cols-1] <<   std::endl;
	res <<  errPtr[errlog.cols-1] <<   std::endl;
	
	out.close();
	return res.str();
}
