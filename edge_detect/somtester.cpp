#include "somtester.h"

#include "myCanny.h"

SOMTester::SOMTester(const SOM &map, const multi_img &img, const EdgeDetectionConfig &conf)
    : som(map), image(img), config(conf),
      lookup(std::vector<std::vector<cv::Point> >(image.height,
                                            std::vector<cv::Point>(image.width)))
{
	for (int y = 0; y < image.height; y++) {
		for (int x = 0; x < image.width; x++) {
			lookup[y][x] = som.identifyWinnerNeuron(image(y, x));
		}
	}
}

cv::Mat SOMTester::generateRankImage() { // TODO: remove operation on m_img_rank (see below)

	rankmap = cv::Mat1f(m_msi.height, m_msi.width);
	float normidx = 1.f / (float)(m_widthSom*m_heightSom);

	bool indirect = !m_rank.empty();

	for (int y = 0; y < m_msi.height; y++) {
		float *row = rankmap[y];
		for (int x = 0; x < m_msi.width; x++) {
			const cv::Point &p = lookup[y][x];
			if (indirect)
				row[x] = m_rank(p) * normidx;
			else
				row[x] = p.x * normidx;
		}
	}


	double min, max;
	cv::minMaxLoc(rankmap, &min, &max);
	std::cerr << "rank image: [" << min << ", " << max << "]" << std::endl;

	cv::imwrite(config.output_dir + "/" + getFilenameExtension()+"_rank.png", rankmap*255.f);

	if(config.isGraphical) {
		compareMultispectralData();
	}

	return rankmap;

}

cv::Mat SOMTester::generateRankImage(cv::Mat_<unsigned int> &rankMatrix) {
  m_rank = rankMatrix; // TODO: wtf why store as member?
	cv::imwrite(config.output_dir + "/" + getFilenameExtension()+"_rankmatrix.png", m_rank);

  return generateRankImage();
}

void SOMTester::getEdge( cv::Mat1d &dx, cv::Mat1d &dy, int mode)
{
	std::cout << "# Setting up neuron lookup table" << std::endl;
	// actually oart of the training, instead of edge image
	m_som.generateLookupTable(m_msi);
	std::cout << "# Done!" << std::endl;
	std::cout << "# Calculating derivatives (dx, dy )" << std::endl;

	dx = cv::Mat::zeros(m_msi.height, m_msi.width, CV_64F);
  dy = cv::Mat::zeros(m_msi.height, m_msi.width, CV_64F);

	//filter coefficients
	double c1,c2,c3;

	cv::Point point;

	double distX,distY;
	distX = 0.0;
	distY = 0.0;

	double p1,p2;
	p1 = 0.0;
	p2 = 0.0;

	double fraction = 1.0;

	//use Sobel mask
	if(mode == 0)
	{
		c1 = 1.0;
		c2 = 2.0;
		c3 = 1.0;
	}
	else //use Scharr mask, expermental
	{
		c1 = 3.0;
		c2 = 10.0;
		c3 = 3.0;
	}

	fraction = (1.0/(c1+c2+c3));

	cv::Mat2d indices(m_msi.height, m_msi.width);
	for (int y = 0; y < m_msi.height ; y++)
	{
		cv::Vec2d *drow = indices[y];
		for (int x = 0; x < m_msi.width ; x++)
		{
			const cv::Point &p = lookup[y][x];
			drow[x][0] = p.x;
			drow[x][1] = p.y;
		}
	}

	double valx,valy;
	bool periodic;
	if(config.graph_type == "MESH_P")
		periodic = true;
	else
		periodic = false;

	double maxIntensity = 0.0;

	unsigned int ten = (m_msi.height* m_msi.width)/10;
	int round = 1;

	if(config.verbosity > 0)
		std::cout << "  0 %" <<std::endl;
	for (int y = 1; y < m_msi.height-1; y++)
	{
		double* x_ptr = dx[y];
		double* y_ptr = dy[y];
		cv::Vec2d *i_ptr = indices[y];
		cv::Vec2d *i_uptr = indices[y-1];
		cv::Vec2d *i_dptr = indices[y+1];
		double xx, yy;

		for (int x = 1; x < m_msi.width-1; x++)
		{
			if(( (y*m_msi.width + x )% ten) == 0 && config.verbosity > 0)
			{
				std::cout << " " << round * 10 << " %" <<std::endl;
				round++;
			}
			{	// y-direction
				cv::Point2d u,d;

				xx = (c1 * i_uptr[x-1][0] + c2 * i_uptr[x][0] + c3 * i_uptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_uptr[x][1] + c3 * i_uptr[x+1][1]) * fraction;

				u.x = xx;
				u.y = yy;
				xx = (c1 * i_dptr[x-1][0] + c2 * i_dptr[x][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_dptr[x-1][1] + c2 * i_dptr[x][1] + c3 * i_dptr[x+1][1]) * fraction;

				d.x = xx;
				d.y = yy;

				if(m_withGraph || m_withUMap)
					valy = graphDistance(u, d);
				else
					valy = vectorDistance(u, d);

				if (maxIntensity < valy)
					maxIntensity = valy;
				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y) )
					valy = -valy;
				y_ptr[x] =  valy;
			}
			{	// x-direction
				cv::Point2d u,d;

				xx = (c1 * i_uptr[x-1][0] + c2 * i_ptr[x-1][0] + c3 * i_dptr[x-1][0]) * fraction;
				yy = (c1 * i_uptr[x-1][1] + c2 * i_ptr[x-1][1] + c3 * i_dptr[x-1][1]) * fraction;

				u.x = xx;
				u.y = yy;
				xx = (c1 * i_uptr[x+1][0] + c2 * i_ptr[x+1][0] + c3 * i_dptr[x+1][0]) * fraction;
				yy = (c1 * i_uptr[x+1][1] + c2 * i_ptr[x+1][1] + c3 * i_dptr[x+1][1]) * fraction;

				d.x = xx;
				d.y = yy;

				if(m_withGraph || m_withUMap)
					valx = graphDistance(u, d);
				else
					valx = vectorDistance(u, d);

				if (maxIntensity < valx)
					maxIntensity = valx;

				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y) )
					valx = -valx;
				x_ptr[x] = valx;
			}
		}
	}
  //normalization
	for (int y = 0; y < m_msi.height ; y++)
	{
		double* x_ptr = dx[y];
		double* y_ptr = dy[y];

		for(int x = 0; x < m_msi.width ; x++)
		{
			x_ptr[x] = ( ((x_ptr[x] + maxIntensity)*0.5/maxIntensity));
			y_ptr[x] =  ( ((y_ptr[x] + maxIntensity)*0.5/maxIntensity));
		}
  }
  if(config.verbosity > 0)
		std::cout << "100 %" <<std::endl;
}

cv::Mat SOMTester::generateEdgeImage(double h1, double h2)
{

	edgemap = cv::Mat_<uchar>(rankmap.size());
	cv::Mat_<uchar> edgeShow;

	for(int y = 0; y < rankmap.rows; y++)
	{
		for(int x = 0; x < rankmap.cols; x++)
		{
			edgemap.at<uchar>(y,x) = static_cast<uchar>(rankmap.at<float>(y,x) *255.0f );
		}
	}

	cv::Canny( edgemap, edgeShow, h1, h2, 3, true );

	cv::imwrite(config.output_dir+getFilenameExtension()+"_edge.png", edgeShow);

	return edgemap;
}
