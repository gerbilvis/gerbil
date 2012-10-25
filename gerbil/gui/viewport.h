/*
	Copyright(c) 2010 Johannes Jordan <johannes.jordan@cs.fau.de>.
	Copyright(c) 2012 Petr Koupy <petr.koupy@gmail.com>

	This file may be licensed under the terms of of the GNU General Public
	License, version 3, as published by the Free Software Foundation. You can
	find it here: http://www.gnu.org/licenses/gpl.html
*/

#ifndef VIEWPORT_H
#define VIEWPORT_H

// Boost multi_array has portability problems (at least for Boost 1.51 and below).
#ifndef __GNUC__
#pragma warning(disable:4996) // disable MSVC Checked Iterators warnings
#endif
#ifndef Q_MOC_RUN
#include <boost/multi_array.hpp> // ensure that multi_array is not visible to Qt MOC
#endif

inline size_t tbb_size_t_select(unsigned u, unsigned long long ull) {
	return (sizeof(size_t) == sizeof(u)) ? size_t(u) : size_t(ull);
}
static const size_t tbb_hash_multiplier = tbb_size_t_select(2654435769U, 11400714819323198485ULL);

namespace tbb {

template<typename T>
inline size_t tbb_hasher(const boost::multi_array<T, 1> &a) {
	size_t h = 0;
	for (size_t i = 0; i < a.size(); ++i)
		h = static_cast<size_t>(a[i]) ^ (h * tbb_hash_multiplier);
	return h;
}

}

#include <multi_img.h>
#include <shared_data.h>
#include <QGLWidget>
#include <QGLBuffer>
#include <QGLFramebufferObject>
#include <vector>
#include <QLabel>
#include <QTimer>
#include <limits>
#include <tbb/atomic.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/task.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/partitioner.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb_allocator.h>

struct Bin {
	Bin() : weight(0.f) {}
	Bin(const multi_img::Pixel& initial_means)
		: weight(1.f), means(initial_means) {} //, points(initial_means.size()) {}

	inline void add(const multi_img::Pixel& p) {
		weight += 1.f;
		if (means.empty())
			means.resize(p.size(), 0.f);
		std::transform(means.begin(), means.end(), p.begin(), means.begin(),
					   std::plus<multi_img::Value>());
	}

	inline void sub(const multi_img::Pixel& p) {
		weight -= 1.f;
		assert(!means.empty());
		std::transform(means.begin(), means.end(), p.begin(), means.begin(),
					   std::minus<multi_img::Value>());
	}

	float weight;
	std::vector<multi_img::Value> means;
	QColor rgb;
	//QPolygonF points;
};

struct BinSet {
	BinSet(const QColor &c, int size)
		: label(c), boundary(size, std::make_pair((int)255, (int)0)) { totalweight = 0; }
	QColor label;
	typedef boost::multi_array<unsigned char, 1> HashKey;
	typedef tbb::concurrent_hash_map<HashKey, Bin> HashMap;
	HashMap bins;
	tbb::atomic<int> totalweight;
	std::vector<std::pair<int, int> > boundary;
};

typedef boost::shared_ptr<SharedData<std::vector<BinSet> > > sets_ptr;

enum representation {
	IMG = 0,
	GRAD = 1,
	IMGPCA = 2,
	GRADPCA = 3,
	REPSIZE = 4
};

struct ViewportCtx {
	ViewportCtx &operator=(const ViewportCtx &other) {
		wait = other.wait;
		reset = other.reset;
		dimensionality = other.dimensionality;
		dimensionalityValid = other.dimensionalityValid;
		type = other.type;
		meta = other.meta;
		metaValid = other.metaValid;
		labels = other.labels;
		labelsValid = other.labelsValid;
		ignoreLabels = other.ignoreLabels;
		nbins = other.nbins;
		binsize = other.binsize;
		binsizeValid = other.binsizeValid;
		minval = other.minval;
		minvalValid = other.minvalValid;
		maxval = other.maxval;
		maxvalValid = other.maxvalValid;
		return *this;
	}

	tbb::atomic<int> wait;
	tbb::atomic<int> reset;
	size_t dimensionality;
	bool dimensionalityValid;
	representation type;
	std::vector<multi_img::BandDesc> meta;
	bool metaValid;
	std::vector<QString> labels;
	bool labelsValid;
	bool ignoreLabels;
	int nbins;
	multi_img::Value binsize;
	bool binsizeValid;
	multi_img::Value minval;
	bool minvalValid;
	multi_img::Value maxval;
	bool maxvalValid;
};

typedef boost::shared_ptr<SharedData<ViewportCtx> > vpctx_ptr;

class Viewport : public QGLWidget
{
	Q_OBJECT
public:
	Viewport(QWidget *parent = 0);
	~Viewport();

	void prepareLines();
	void setLimiters(int label);

	vpctx_ptr ctx;
	sets_ptr sets;

	QGLBuffer vb;
	tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > shuffleIdx;

	bool illuminant_correction;
	std::vector<multi_img::Value> illuminant;

	int selection, hover;
	bool limiterMode;
	std::vector<std::pair<int, int> > limiters;
	bool active, wasActive;

	float useralpha;

	bool showLabeled, showUnlabeled;
	bool overlayMode;
	QPolygonF overlayPoints;

	enum RenderMode {
		RM_SKIP = 0,
		RM_STEP = 1,
		RM_FULL = 2
	};

public slots:
	void killHover();
	void toggleRGB(bool enabled)
	{ drawRGB = enabled; updateTextures(); }
	void activate();

	// entry and exit point of user interaction with quick drawing
	void startNoHQ(bool resize = false);
	bool endNoHQ();
	void resizeEpilog();

	// acknowledge folding
	void folding() { drawingState = FOLDING; resizeTimer.start(50); }

	void screenshot();

	void rebuild();

	void updateTextures(RenderMode spectrum = RM_STEP, RenderMode highlight = RM_STEP);

protected slots:
	void continueDrawingSpectrum();
	void continueDrawingHighlight();

signals:
	void bandSelected(representation type, int dim);
	void newOverlay(int dim);
	void activated();
	void addSelection();
	void remSelection();

protected:
	void reset();
	void paintEvent(QPaintEvent*);
	void resizeEvent(QResizeEvent*);
	void enterEvent(QEvent*);
	void mouseMoveEvent(QMouseEvent*);
	void mousePressEvent(QMouseEvent*);
	void mouseReleaseEvent(QMouseEvent*);
	void wheelEvent(QWheelEvent *);
	void keyPressEvent(QKeyEvent *);

	// helper function that updates Y-axis labels
	void updateYAxis();

	// helper function that updates world transformation
	void updateModelview();

	// helper functions called by mouseMoveEvent
	void updateXY(int sel, int bin);

	// helper functions called by paintEvent
	void drawBins(QPainter &painter, QTimer &renderTimer, 
		unsigned int &renderedLines, unsigned int renderStep, bool onlyHighlight);
	void drawAxesBg(QPainter&);
	void drawAxesFg(QPainter&);
	void drawLegend(QPainter&);
	void drawOverlay(QPainter &);
	void drawWaitMessage(QPainter&);

	// helper for limiter handling
	bool updateLimiter(int dim, int bin);

	class PreprocessBins {
	public:
		PreprocessBins(int label, size_t dimensionality, multi_img::Value maxval, 
			std::vector<multi_img::BandDesc> &meta, 
			tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > &shuffleIdx) 
			: label(label), dimensionality(dimensionality), maxval(maxval), meta(meta),
			shuffleIdx(shuffleIdx), ranges(dimensionality, std::pair<int, int>(INT_MAX, INT_MIN)) {}
		PreprocessBins(PreprocessBins &toSplit, tbb::split) 
			: label(toSplit.label), dimensionality(toSplit.dimensionality), 
			maxval(toSplit.maxval), meta(toSplit.meta),
			shuffleIdx(toSplit.shuffleIdx), ranges(dimensionality, std::pair<int, int>(INT_MAX, INT_MIN)) {} 
		void operator()(const BinSet::HashMap::range_type &r);
		void join(PreprocessBins &toJoin);
		std::vector<std::pair<int, int> > GetRanges() { return ranges; }
	private:
		int label;
		size_t dimensionality;
		multi_img::Value maxval;
		std::vector<multi_img::BandDesc> &meta;
		tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > &shuffleIdx;
		std::vector<std::pair<int, int> > ranges;
	};

	class GenerateVertices {
	public:
		GenerateVertices(bool drawMeans, size_t dimensionality, multi_img::Value minval, multi_img::Value binsize,
			bool illuminant_correction, std::vector<multi_img::Value> &illuminant, std::vector<BinSet> &sets,
			tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > &shuffleIdx, GLfloat *varr) 
			: drawMeans(drawMeans), dimensionality(dimensionality), minval(minval), binsize(binsize),
			illuminant_correction(illuminant_correction), illuminant(illuminant), sets(sets),
			shuffleIdx(shuffleIdx), varr(varr) {}
		void operator()(const tbb::blocked_range<size_t> &r) const;
	private:
		bool drawMeans;
		size_t dimensionality;
		multi_img::Value minval;
		multi_img::Value binsize;
		bool illuminant_correction;
		std::vector<multi_img::Value> &illuminant;
		std::vector<BinSet> &sets;
		tbb::concurrent_vector<std::pair<int, BinSet::HashKey> > &shuffleIdx;
		GLfloat *varr;
	};

private:
	QGLFramebufferObject *fboSpectrum;
	QGLFramebufferObject *fboHighlight;
	QGLFramebufferObject *fboMultisamplingBlit;

	// modelview matrix and its inverse
	QTransform modelview, modelviewI;
	// zoom and shift in y-direction
	qreal zoom;
	qreal shift;
	int lasty;

	/* if in limiter mode, user has to release mouse button before switching
	   band. this is for usability, users tend to accidentially switch bands */
	bool holdSelection;
	int *activeLimiter;

	// cache for efficient overlay
	bool cacheValid;
	QImage cacheImg;

	// draw without highlight
	bool clearView;
	bool implicitClearView;

	// drawing mode mean vs. bin center
	bool drawMeans;
	// drawing mode labelcolor vs. sRGB color
	bool drawRGB;
	// draw with antialiasing
	bool drawHQ;

	enum {
		HIGH_QUALITY,        // drawing HQ as usual
		HIGH_QUALITY_QUICK,  // last update was HQ, quick updates requested
		QUICK,               // last update not HQ, quick updates requested
		RESIZE,              // resize updates requested (last update invalid)
		SCREENSHOT,          // screenshot update requested (special drawing)
		FOLDING              // only draw blank during folding resize ops
	} drawingState;
	// this timer will re-enable regular drawing after resize/folding
	QTimer resizeTimer;
	QTimer scrollTimer;

	static const unsigned int renderAtOnceStep = 1024 * 1024 * 1024;

	static const unsigned int spectrumRenderStep = 10000;
	QTimer spectrumRenderTimer;
	unsigned int spectrumRenderedLines;

	static const unsigned int highlightRenderStep = 10000;
	QTimer  highlightRenderTimer;
	unsigned int  highlightRenderedLines;

	std::vector<QString> yaxis;
	int yaxisWidth;
};

#endif // VIEWPORT_H
