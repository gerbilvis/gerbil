#include "compute.h"

#include "../gerbil_gui_debug.h"

#include <QGLBuffer>

// altmann, debugging helper function
void assertBinSetsKeyDim(const std::vector<BinSet> &v, const ViewportCtx &ctx) {
	using namespace std;
	assert(v.size() > 0);

	foreach(BinSet set, v) {
		foreach(BinSet::HashMap::value_type pr, set.bins) {
			const BinSet::HashKey &key = pr.first;
			if (ctx.dimensionality != key.size()) {
				GGDBGP(boost::format("failure: type=%1% ,  (key.size()==%2%  != dim==%3%)")
				   %ctx.type %key.size() %ctx.dimensionality
				   << endl);
				assert(ctx.dimensionality == key.size());
			}
		}
	}
}

/* translate image value to value in binning coordinate system */
multi_img::Value Compute::curpos(
	const multi_img::Value& val, int dim,
	const multi_img::Value& minval, const multi_img::Value& binsize,
	const std::vector<multi_img::Value> &illuminant)
{
	multi_img::Value curpos = (val - minval) / binsize;
	if (!illuminant.empty())
		curpos /= illuminant[dim];
	return curpos;
}

void Compute::PreprocessBins::operator()(const BinSet::HashMap::range_type &r)
{
	cv::Vec3f color;
	multi_img::Pixel pixel(dimensionality);
	BinSet::HashMap::iterator it;
	for (it = r.begin(); it != r.end(); it++) {
		Bin &b = it->second;
		for (int d = 0; d < dimensionality; ++d) {
			pixel[d] = b.means[d] / b.weight;
			std::pair<int, int> &range = ranges[d];
			range.first = std::min<int>(range.first, (int)(it->first)[d]);
			range.second = std::max<int>(range.second, (int)(it->first)[d]);
		}
		// TODO: calculate colors for all pixels BEFORE this step with functor
		color = multi_img::bgr(pixel, dimensionality, meta, maxval);
		b.rgb = QColor(color[2]*255, color[1]*255, color[0]*255);
		index.push_back(make_pair(label, it->first));
	}
}

void Compute::PreprocessBins::join(PreprocessBins &toJoin)
{
	for (int d = 0; d < dimensionality; ++d) {
		std::pair<int, int> &local = ranges[d];
		std::pair<int, int> &remote = toJoin.ranges[d];
		local.first = std::min<int>(local.first, remote.first);
		local.second = std::max<int>(local.second, remote.second);
	}
}

void Compute::preparePolylines(const ViewportCtx &ctx,
							   std::vector<BinSet> &sets, binindex &index)
{
	assertBinSetsKeyDim(sets, ctx);

	assert(sets.size()>0);
	index.clear();
	for (unsigned int i = 0; i < sets.size(); ++i) {
		BinSet &s = sets[i];
		PreprocessBins preprocess(i, ctx.dimensionality,
			ctx.maxval, ctx.meta, index);
		tbb::parallel_reduce(BinSet::HashMap::range_type(s.bins),
			preprocess, tbb::auto_partitioner());
		s.boundary = preprocess.GetRanges();
	}

	assert(index.begin() < index.end()); // index is not empty

	// shuffle the index for clutter-reduction
	std::random_shuffle(index.begin(), index.end());
}

int Compute::storeVertices(const ViewportCtx &ctx,
						   const std::vector<BinSet> &sets,
						   const binindex& index, QGLBuffer &vb,
						   bool drawMeans,
						   const std::vector<multi_img::Value> &illuminant)
{
	vb.setUsagePattern(QGLBuffer::StaticDraw);
	bool success = vb.create();
	if (!success)
		return -1;

	success = vb.bind();
	if (!success)
		return 1;

	//GGDBGM(boost::format("shuffleIdx.size()=%1%, (*ctx)->dimensionality=%2%\n")
	//	   %shuffleIdx.size() %(*ctx)->dimensionality)
	vb.allocate(index.size() * ctx.dimensionality * sizeof(GLfloat) * 2);
	//GGDBGM("before vb.map()\n");
	GLfloat *varr = (GLfloat*)vb.map(QGLBuffer::WriteOnly);
	//GGDBGM("after vb.map()\n");

	if (!varr)
		return 2;

	GenerateVertices generate(drawMeans, ctx.dimensionality, ctx.minval,
							  ctx.binsize, illuminant, sets, index, varr);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, index.size()),
		generate, tbb::auto_partitioner());

	vb.unmap();
	vb.release();
	return 0;
}

void Compute::GenerateVertices::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (size_t i = r.begin(); i != r.end(); ++i) {
		const std::pair<int, BinSet::HashKey> &idx = index[i];
		const BinSet &s = sets[idx.first];
		const BinSet::HashKey &K = idx.second;
		const Bin &b = s.bins.equal_range(K).first->second;
		int vidx = i * 2 * dimensionality;
		for (int d = 0; d < dimensionality; ++d) {
			qreal curpos;
			if (drawMeans) {
				curpos = ((b.means[d] / b.weight) - minval) / binsize;
			} else {
				curpos = (unsigned char)K[d] + 0.5;
				if (!illuminant.empty())
					curpos *= illuminant[d];
			}
			varr[vidx++] = d;
			varr[vidx++] = curpos;
		}
	}
}
