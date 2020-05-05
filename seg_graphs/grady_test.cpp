extern "C" {
	#include "csparse/cs.h"
}

void printmat(const cv::Mat1f &X)
{
	for (int y = 0; y < X.rows; ++y) {
		for (int x = 0; x < X.cols; ++x) {
			std::cout << (x > 0 ? ", " : "") << X(y, x);
		}
		std::cout << "; ";
	}
	std::cout << std::endl;
}

void ipp(const multi_img& input)
{
	input.rebuildPixels(false);
	int nodes = input.width*input.height, width = input.width;
	cs *Z2, *Z;
	cs *D2, *D;
	cs *L2, *L;
	Z2 = cs_spalloc(input.size(), nodes, input.size()*nodes, 1, 1);
	D2 = cs_spalloc(nodes, nodes, nodes, 1, 1);
	L2 = cs_spalloc(nodes, nodes, nodes*5, 1, 1);
	int z = 0, n = 0, l = 0;
	for (int y = 0; y < input.height; ++y) {
		for (int x = 0; x < input.width; ++x, ++n, ++l) {
			const multi_img::Pixel &p = input(y, x);
			const cv::Mat1f v(p, false);

			// fill Z
			for (int k = 0; k < p.size(); ++k, ++z) {
				Z2->x[z] = p[k];
				Z2->i[z] = k;
				Z2->p[z] = n;
			}

			// fill D and L
			float degree = 0.f;
			if (y > 0) {
				cv::Mat1f v2(input(y-1, x), false);
				float w = cv::norm(v, v2, cv::NORM_L2);
				degree += w;
				L2->x[l] = -w;
				L2->i[l] = y*width + x;
				L2->p[l] = (y-1)*width + x;
				l++;
			}
			if (x > 0) {
				cv::Mat1f v2(input(y, x-1), false);
				float w = cv::norm(v, v2, cv::NORM_L2);
				degree += w;
				L2->x[l] = -w;
				L2->i[l] = y*width + x;
				L2->p[l] = y*width + x-1;
				l++;
			}
			if (y < input.height - 1) {
				cv::Mat1f v2(input(y+1, x), false);
				float w = cv::norm(v, v2, cv::NORM_L2);
				degree += w;
				L2->x[l] = -w;
				L2->i[l] = y*width + x;
				L2->p[l] = (y+1)*width + x;
				l++;
			}
			if (x < input.width - 1) {
				cv::Mat1f v2(input(y, x+1), false);
				float w = cv::norm(v, v2, cv::NORM_L2);
				degree += w;
				L2->x[l] = -w;
				L2->i[l] = y*width + x;
				L2->p[l] = y*width + x+1;
				l++;
			}
			D2->x[n] = degree;
			D2->i[n] = n;
			D2->p[n] = n;

			L2->x[l] = degree;
			L2->i[l] = n;
			L2->p[l] = n;
		}
	}
	Z2->nz = z;
	Z2->m  = input.size();
	Z2->n  = nodes;
	D2->nz = n;
	D2->m  = nodes;
	D2->n  = nodes;
	L2->nz = l;
	L2->m  = nodes;
	L2->n  = nodes;

	Z = cs_compress(Z2);
	cs_spfree(Z2);
	D = cs_compress(D2);
	cs_spfree(D2);
	L = cs_compress(L2);
	cs_spfree(L2);
	cs *ZT;
	ZT = cs_transpose(Z, 1);

	// right side
	cs *DZT_tmp, *ZDZT;
	DZT_tmp = cs_multiply(D, ZT);
	ZDZT = cs_multiply(Z, DZT_tmp);
	cs_spfree(D);	cs_spfree(DZT_tmp);
	cs_print(ZDZT, 1);

	// left side
	cs *LZT_tmp, *ZLZT;
	LZT_tmp = cs_multiply(L, ZT);
	ZLZT = cs_multiply(Z, LZT_tmp);
	cs_spfree(L);	cs_spfree(LZT_tmp);
	cs_print(ZLZT, 1);
	cs_spfree(Z);	cs_spfree(ZT);

	cv::Mat1f A(input.size(), input.size(), 0.f);
	cv::Mat1f B = A.clone();
	for (int j = 0; j < ZLZT->n; ++j) {
		int i = 0;
		for (int p = ZLZT->p[j]; p < ZLZT->p[j+1]; ++p, ++i) {
			A(j, i) = ZLZT->x[p];
		}
	}
	for (int j = 0; j < ZDZT->n; ++j) {
		int i = 0;
		for (int p = ZDZT->p[j]; p < ZDZT->p[j+1]; ++p, ++i) {
			B(j, i) = ZDZT->x[p];
		}
	}
	cs_spfree(ZLZT);	cs_spfree(ZDZT);
	cv::Mat1f X = B.inv() * A;
	printmat(X);
	cv::SVD likeaboss(X);
	printmat(likeaboss.u);
	printmat(likeaboss.vt.t());
}

