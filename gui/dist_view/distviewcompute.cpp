#include "distviewcompute.h"

#include <gerbilapplication.h>

//#define GGDBG_MODULE
#include "../gerbil_gui_debug.h"

#include <QGLBuffer>

// altmann, debugging helper function
bool assertBinSetsKeyDim(const std::vector<BinSet> &v, const ViewportCtx &ctx) {
	assert(v.size() > 0);

	foreach(BinSet set, v) {
		foreach(BinSet::HashMap::value_type pr, set.bins) {
			const BinSet::HashKey &key = pr.first;
			if (ctx.dimensionality != key.size()) {
				GGDBGP(boost::format("failure: type=%1% ,  (key.size()==%2%  != dim==%3%)")
				   %ctx.type %key.size() %ctx.dimensionality
				   << std::endl);
				return false;
			}
		}
	}
	return true;
}

/** RAII class to manage QGLBuffer. */
class GLBufferHolder
{
public:
	GLBufferHolder(QGLBuffer &vb)
		: vb(vb)
	{
		const char* pfx = "GLBufferHolder::GLBufferHolder(): ";
		msuccess = vb.create();
		if (!msuccess) {
			std::stringstream err;
			err << pfx << "QGLBuffer::create() failed." << std::endl;
			GerbilApplication::instance()->
				criticalError(QString::fromStdString(err.str()));
			return;
		}
		msuccess = vb.bind();
		if (!msuccess) {
			std::stringstream err;
			err << pfx << "QGLBuffer::bind() failed" << std::endl;
			GerbilApplication::instance()->
				criticalError(QString::fromStdString(err.str()));
			return;
		}
	}

	bool success() {
		return msuccess;
	}

	~GLBufferHolder() {
		vb.unmap();
		vb.release();
	}


private:
	bool msuccess;
	QGLBuffer &vb;
};


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
	BinSet::HashMap::iterator it;
	for (it = r.begin(); it != r.end(); it++) {
		Bin &b = it->second;
		for (int d = 0; d < dimensionality; ++d) {
			std::pair<int, int> &range = ranges[d];
			range.first = std::min<int>(range.first, (int)(it->first)[d]);
			range.second = std::max<int>(range.second, (int)(it->first)[d]);
		}
		b.rgb = QColor(b.color[2]/b.weight, b.color[1]/b.weight, b.color[0]/b.weight);
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
	if (!assertBinSetsKeyDim(sets, ctx)) {
		return;
	}

	assert(sets.size()>0);
	index.clear();
	//GGDBGP("Compute::preparePolylines() sets.size() = " << sets.size() << endl);
	for (unsigned int i = 0; i < sets.size(); ++i) {
		BinSet &s = sets[i];
		PreprocessBins preprocess(i, ctx.dimensionality,
			ctx.maxval, ctx.meta, index);
		tbb::parallel_reduce(BinSet::HashMap::range_type(s.bins),
			preprocess, tbb::auto_partitioner());
		s.boundary = preprocess.GetRanges();
	}

	if (index.begin() == index.end()) {
		GGDBGP("Compute::preparePolylines(): error: empty index" << endl);
		return;
	}

	// shuffle the index for clutter-reduction
	std::random_shuffle(index.begin(), index.end());
}

void Compute::storeVertices(const ViewportCtx &ctx,
						   const std::vector<BinSet> &sets,
						   const binindex& index, QGLBuffer &vb,
						   bool drawMeans,
						   const std::vector<multi_img::Value> &illuminant)
{
	vb.setUsagePattern(QGLBuffer::StaticDraw);
	GLBufferHolder vbh(vb);
	if (!vbh.success()) {
		return;
	}

	//GGDBGM(boost::format("shuffleIdx.size()=%1%, (*ctx)->dimensionality=%2%\n")
	//	   %shuffleIdx.size() %(*ctx)->dimensionality)
	if (index.size() == 0) {
		std::cerr << "Compute::storeVertices(): error: empty binindex"
				  << std::endl;
		return;
	}
	const size_t nbytes = index.size() *
			ctx.dimensionality * sizeof(GLfloat) * 2;
	//GGDBGP("Compute::storeVertices(): allocating "<< nbytes << " bytes" << endl);
	vb.allocate(nbytes);
	//GGDBGM("before vb.map()\n");
	GLfloat *varr = (GLfloat*)vb.map(QGLBuffer::WriteOnly);
	//GGDBGM("after vb.map()\n");

	if (!varr) {
		std::stringstream err;
		err << "Compute::storeVertices(): QGLBuffer::map() failed" << std::endl;
		GerbilApplication::instance()->
			criticalError(QString::fromStdString(err.str()));
		return;
	}

	GenerateVertices generate(drawMeans, ctx.dimensionality, ctx.minval,
							  ctx.binsize, illuminant, sets, index, varr);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, index.size()),
		generate, tbb::auto_partitioner());

	return;
}

void Compute::GenerateVertices::operator()(const tbb::blocked_range<size_t> &r) const
{
	for (tbb::blocked_range<size_t>::const_iterator i = r.begin();
		 i != r.end();
		 ++i)
	{
		const std::pair<size_t, BinSet::HashKey> &idx = index[i];
		if ( !(0 <= idx.first || idx.first < sets.size())) {
			GGDBGM("bad sets index"<< endl);
			return;
		}
		const BinSet &s = sets[idx.first];
		const BinSet::HashKey &K = idx.second;
		// concurrent_hash_map::equal_range may not be used for concurrent
		// access.
		// See http://www.threadingbuildingblocks.org/docs/help/reference/containers_overview/concurrent_hash_map_cls.htm
		// However, this is part of a read-only iteration, where the BinSet is
		// never modified. Thus using concurrent_hash_map::equal_range seems
		// OK.
		std::pair<BinSet::HashMap::const_iterator,
				  BinSet::HashMap::const_iterator>
				binitp = s.bins.equal_range(K);
		if (s.bins.end() == binitp.first) {
			GGDBGM("no bin"<< endl);
			return;
		}
		const Bin &b = binitp.first->second;
		int vidx = i * 2 * dimensionality;
		for (size_t d = 0; d < dimensionality; ++d) {
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
