#include "somtester.h"

#include "myCanny.h"
#include <cv.h>
#include <highgui.h>

SOMTester::SOMTester(const SOM &map, const multi_img &img, const vole::EdgeDetectionConfig &conf)
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

cv::Mat SOMTester::generateRankImage() {
	rankmap = cv::Mat1f(image.height, image.width);
	float normidx = 1.f / (float)(som.getWidth() * som.getHeight());

	bool indirect = !sfcmap.empty();

	for (int y = 0; y < image.height; y++) {
		float *row = rankmap[y];
		for (int x = 0; x < image.width; x++) {
			const cv::Point &p = lookup[y][x];
			if (indirect)
				row[x] = sfcmap(p) * normidx;
			else
				row[x] = p.x * normidx;
		}
	}


	double min, max;
	cv::minMaxLoc(rankmap, &min, &max);
	std::cerr << "rank image: [" << min << ", " << max << "]" << std::endl;

	cv::imwrite(config.output_dir + "/rank.png", rankmap*255.f);

	return rankmap;

}

cv::Mat SOMTester::generateRankImage(cv::Mat_<unsigned int> &rankMatrix)
{
	sfcmap = rankMatrix; // TODO: wtf why store as member?
	cv::imwrite(config.output_dir + "/rankmatrix.png", sfcmap);

	return generateRankImage();
}

void SOMTester::getEdge(cv::Mat1d &dx, cv::Mat1d &dy)
{
	std::cout << "Calculating derivatives (dx, dy)" << std::endl;

	dx = cv::Mat::zeros(image.height, image.width, CV_64F);
	dy = cv::Mat::zeros(image.height, image.width, CV_64F);

	// collect SOM representant coordinates of all pixels
	cv::Mat2d indices(image.height, image.width);
	for (int y = 0; y < image.height; y++) {
		cv::Vec2d *drow = indices[y];
		for (int x = 0; x < image.width; x++) {
			const cv::Point &p = lookup[y][x];
			drow[x][0] = p.x;
			drow[x][1] = p.y;
		}
	}

	double maxIntensity = 0.0;

	unsigned int ten = (image.height* image.width)/10;
	int round = 1;
	if(config.verbosity > 0)
		std::cout << "  0 %" <<std::endl;

	for (int y = 1; y < image.height-1; y++) {
		double* x_ptr = dx[y];
		double* y_ptr = dy[y];
		cv::Vec2d *i_ptr = indices[y];
		cv::Vec2d *i_uptr = indices[y-1];
		cv::Vec2d *i_dptr = indices[y+1];
		double xx, yy, valx, valy;

		for (int x = 1; x < image.width-1; x++) {
			if (config.verbosity > 0 && ((y*image.width + x) % ten) == 0) {
				std::cout << " " << round * 10 << " %" <<std::endl;
				round++;
			}
			{	// y-direction
				xx = (i_uptr[x-1][0] + 2*i_uptr[x][0] + i_uptr[x+1][0]) * 0.25;
				yy = (i_uptr[x-1][1] + 2*i_uptr[x][1] + i_uptr[x+1][1]) * 0.25;
				cv::Point2f u(xx, yy);
				xx = (i_dptr[x-1][0] + 2*i_dptr[x][0] + i_dptr[x+1][0]) * 0.25;
				yy = (i_dptr[x-1][1] + 2*i_dptr[x][1] + i_dptr[x+1][1]) * 0.25;
				cv::Point2f d(xx, yy);

				valy = som.getDistance(u, d);
				if (maxIntensity < valy)
					maxIntensity = valy;

				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valy = -valy;
				y_ptr[x] =  valy;
			}
			{	// x-direction
				xx = (i_uptr[x-1][0] + 2*i_ptr[x-1][0] + i_dptr[x-1][0]) * 0.25;
				yy = (i_uptr[x-1][1] + 2*i_ptr[x-1][1] + i_dptr[x-1][1]) * 0.25;
				cv::Point2f u(xx, yy);
				xx = (i_uptr[x+1][0] + 2*i_ptr[x+1][0] + i_dptr[x+1][0]) * 0.25;
				yy = (i_uptr[x+1][1] + 2*i_ptr[x+1][1] + i_dptr[x+1][1]) * 0.25;
				cv::Point2f d(xx, yy);

				valx = som.getDistance(u, d);
				if (maxIntensity < valx)
					maxIntensity = valx;

				if ((u.x*u.x + u.y*u.y) < (d.x*d.x + d.y*d.y))
					valx = -valx;
				x_ptr[x] = valx;
			}
		}
	}
	if (config.verbosity > 0)
		std::cout << "100 %" <<std::endl;

	// normalization
	cv::MatIterator_<double> ix, iy;
	for (ix = dx.begin(), iy = dy.begin(); ix != dx.end(); ++ix, ++iy) {
		// [-X .. X] -> [0 .. 1] but not really!! TODO
		*ix = ((*ix + maxIntensity) * 0.5 / maxIntensity);
		*iy = ((*iy + maxIntensity) * 0.5 / maxIntensity);
	}
}

cv::Mat SOMTester::generateEdgeImage(double h1, double h2)
{
	cv::Mat1b edgemap = rankmap * 255.f;
	cv::Mat1b edgeShow;
	cv::Canny(edgemap, edgeShow, h1, h2, 3, true);

	cv::imwrite(config.output_dir + "/edge.png", edgeShow);

	return edgeShow;
}
