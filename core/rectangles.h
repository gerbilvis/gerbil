#ifndef RECTANGLES_H
#define RECTANGLES_H


/** compute complement rectangles
 *  @arg result array of rectangles to fill
 *  @return sum of rectangle areas
 */
int rectComplement(int width, int height, cv::Rect r,
				   std::vector<cv::Rect> &result);

/** compute (A | B) - (A & B) and provide intersecting regions
 *  in coordinates relative to A and B
 *  @arg sub result array of A \ B (in A coords)
 *  @arg add result array of B \ A (in B coords)
 *  @return (A | B) - (A & B) < B
 */
bool rectTransform(const cv::Rect &oldR, const cv::Rect &newR,
				   std::vector<cv::Rect> &sub,
				   std::vector<cv::Rect> &add);

#endif // RECTANGLES_H
