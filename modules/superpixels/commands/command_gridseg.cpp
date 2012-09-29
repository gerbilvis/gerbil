#include "command_gridseg.h"
#include "superpixel_segmentation.h"
#include "color.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace vole {

CommandGridSeg::CommandGridSeg():
	Command("gridseg",
		config,
		"Christian Riess",
		"christian.riess@informatik.uni-erlangen.de"
	)
{
}

int CommandGridSeg::execute()
{
	if (((config.block_size < 0) && (config.number_blocks < 0)) || ((config.block_size > 0) && (config.number_blocks > 0))) {
		std::cerr << "arguments --block_size and --number_blocks are inconsistent: Exactly one of them needs to be positive (to be used), the other one negative (to be ignored), aborted.";
		return 1;
	}

	cv::Mat_<cv::Vec3b> prior_segmentation;
	if (config.prior_segmentation.length() < 1) {
		if (config.fuse_max_area) {
			std::cerr << "--prior_segmentation must be set if --fuse_max_area is set (base image required). Aborted." << std::endl;
			return 1;
		}
		if ((config.x_dim <= 0) || (config.y_dim <= 0)) {
			std::cerr << "--prior_segmentation is not set -> --x_dim and --ydim must be two positive numbers (the dimensions of the target segmentation), aborted." << std::endl;
			return 1;
		}
		prior_segmentation = cv::Mat_<cv::Vec3b>(config.y_dim, config.x_dim, cv::Vec3b(0,0,0));
	} else {
		prior_segmentation = cv::imread(config.prior_segmentation);
		if (prior_segmentation.rows == 0) {
			std::cerr << "unable to read file in --prior_segmentation (" << config.prior_segmentation << "), aborted." << std::endl;
			return 1;
		}
		config.x_dim = prior_segmentation.cols;
		config.y_dim = prior_segmentation.rows;
	}

	if (config.output_file.length() < 1) {
		std::cerr << "no file name for final segmentation given (--output_file), aborted." << std::endl;
	}


//	std::vector<superpixels::Superpixel> superpixels;
//	superpixels::SuperpixelSegmentation::imageToSuperpixels(prior_segmentation, superpixels)

	int step_size_x = config.block_size;
	int step_size_y = config.block_size;
	if (step_size_x < 0) {
		step_size_x = config.x_dim / config.number_blocks;
		step_size_y = config.y_dim / config.number_blocks;
	}
	int n_blocks_x = std::ceil((double)config.x_dim / (double)step_size_x);
	int n_blocks_y = std::ceil((double)config.y_dim / (double)step_size_y);

	if (config.fuse_max_area) {
		cv::Mat_<cv::Vec3b> mosaick = fuse_by_segmentation(prior_segmentation, step_size_x, step_size_y, n_blocks_x, n_blocks_y);
		cv::imwrite(config.output_file, mosaick);
		return 0;
	}

	cv::Mat_<cv::Vec3b> superpixelsImage(prior_segmentation.rows, prior_segmentation.cols, cv::Vec3b(255,255,255));
	int seg_count = 0;
	cv::Vec3b seg_col;
	cv::Vec3b white(255,255,255);

	for (int y = 0; y < n_blocks_y; ++y) {
		for (int x = 0; x < n_blocks_x; ++x) {
			int yp = y*step_size_y;
			int xp = x*step_size_x;
			cv::Vec3b next_col = prior_segmentation[yp][xp];
			bool unvisited_colors = true;
			while (unvisited_colors) {
				unvisited_colors = false;
				cv::Vec3b cur_col = next_col;
				rbase::Color::idToRGB(seg_count, seg_col[2], seg_col[1], seg_col[0]);
				for (int yy = 0; ((yy < step_size_y) && (yp+yy < prior_segmentation.rows)); ++yy) {
					for (int xx = 0; ((xx < step_size_x) && (xp+xx < prior_segmentation.cols)); ++xx) {
						if ((prior_segmentation[yp+yy][xp+xx][0] == white[0]) && (prior_segmentation[yp+yy][xp+xx][1] == white[1]) && (prior_segmentation[yp+yy][xp+xx][2] == white[2])) {
							continue; // visited
						}
						if ((prior_segmentation[yp+yy][xp+xx][0] == cur_col[0]) && (prior_segmentation[yp+yy][xp+xx][1] == cur_col[1]) && (prior_segmentation[yp+yy][xp+xx][2] == cur_col[2])) {
							superpixelsImage[yp+yy][xp+xx] = seg_col; // mark as belonging to the current segment
							prior_segmentation[yp+yy][xp+xx] = white; // mark as used
							continue;
						}
						next_col = prior_segmentation[yp+yy][xp+xx]; // got so far? then there must be two segments in this grid cell
						unvisited_colors = true;
					}
				}
				seg_count++;
			}
		}
	}

	if (config.output_file.length() > 0) {
		cv::imwrite(config.output_file, superpixelsImage);
	}

	return 0;
}

cv::Mat_<cv::Vec3b> CommandGridSeg::fuse_by_segmentation(cv::Mat_<cv::Vec3b> prior_segmentation, int step_size_x, int step_size_y, int n_blocks_x, int n_blocks_y)
{
	cv::Mat_<cv::Vec3b> mosaick(prior_segmentation.rows, prior_segmentation.cols);

	// selected dumbest possible solution to count the number of pixels in one block. Improve this if you feel like it.

	for (int y = 0; y < n_blocks_y; ++y) {
		for (int x = 0; x < n_blocks_x; ++x) {
			int yp = y*step_size_y;
			int xp = x*step_size_x;
			std::vector<std::pair<cv::Vec3b, int> > colors;
			
			for (int yy = 0; ((yy < step_size_y) && (yp+yy < prior_segmentation.rows)); ++yy) {
				for (int xx = 0; ((xx < step_size_x) && (xp+xx < prior_segmentation.cols)); ++xx) {
					cv::Vec3b cur_col = prior_segmentation[yp+yy][xp+xx];
					bool found = false;
					for (unsigned int c = 0; c < colors.size(); ++c) {
						if ((cur_col[0] == colors[c].first[0]) && (cur_col[1] == colors[c].first[1]) && (cur_col[2] == colors[c].first[2])) {
							colors[c].second = colors[c].second + 1;
							found = true;
						}
					}
					if (!found) colors.push_back(std::pair<cv::Vec3b, int>(cur_col, 1));
				}
			}
			// find maximum
			cv::Vec3b cell_col(255,255,255);
			int maxval = -1;
			for (unsigned int i = 0; i < colors.size(); ++i) {
				if (colors[i].second > maxval) {
					maxval = colors[i].second;
					cell_col = colors[i].first;
				}
			}
			// repaint cell
			for (int yy = 0; ((yy < step_size_y) && (yp+yy < prior_segmentation.rows)); ++yy) {
				for (int xx = 0; ((xx < step_size_x) && (xp+xx < prior_segmentation.cols)); ++xx) {
					mosaick[yp+yy][xp+xx] = cell_col;
				}
			}
		}
	}
	return mosaick;
}


void CommandGridSeg::printShortHelp() const
{
	std::cout << "Segments the image in a grid, obtionally intersecting it with another segmentation." << std::endl;
}

void CommandGridSeg::printHelp() const
{
	std::cout << "Segments the image in a grid, obtionally intersecting it with another segmentation." << std::endl;
}

} // vole
