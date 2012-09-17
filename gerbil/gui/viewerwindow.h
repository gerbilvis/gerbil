#ifndef VIEWERWINDOW_H
#define VIEWERWINDOW_H

#include "ui_viewerwindow.h"
#include <shared_data.h>
#include <background_task.h>
#include <multi_img.h>
#include <multi_img_tasks.h>
#include <labeling.h>
#include <illuminant.h>
#include <progress_observer.h>
#include "commandrunner.h"
#ifdef WITH_SEG_MEANSHIFT
#include <meanshift_shell.h>
#endif
#ifdef WITH_SEG_MEDIANSHIFT
#include <medianshift_shell.h>
#endif
#ifdef WITH_SEG_PROBSHIFT
#include <probshift_shell.h>
#endif
#include <graphseg.h>

#include <vector>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QMenu>
#include <opencv2/core/core.hpp>

class ViewerWindow : public QMainWindow, private Ui::ViewerWindow {
    Q_OBJECT
public:
	ViewerWindow(multi_img *image, QWidget *parent = 0);

	const QPixmap* getBand(representation type, int dim);
	const inline Illuminant & getIlluminant(int temp);
	const inline std::vector<multi_img::Value> & getIlluminantC(int temp);

	static QIcon colorIcon(const QColor& color);

public slots:
	void reshapeDock(bool floating);
	void selectBand(representation type, int dim);
	void addToLabel()   { labelmask(false); }
	void remFromLabel() { labelmask(true); }
	void setActive(int id); // id mapping see initUI()
	void newOverlay();

	void startGraphseg();

	void startUnsupervisedSeg(bool findKL = false);
	void startFindKL();
	void segmentationFinished();
	void segmentationApply(std::map<std::string, boost::any>);

	void applyIlluminant();
	void setI1(int index);
	void setI1Visible(bool);
	void bandsSliderMoved(int b);

	void usMethodChanged(int idx);
	void usInitMethodChanged(int idx);
	void usBandwidthMethodChanged(const QString &current);
	void unsupervisedSegCancelled();

	void normTargetChanged(bool usecurrent = false);
	void normModeSelected(int mode, bool targetchange = false, bool usecurrent = false);
	void normModeFixed();
	void applyNormUserRange(bool update = true);
	void clampNormUserRange();

	void loadLabeling(QString filename = "");
	void loadSeeds();
	void saveLabeling();
	// add new label (color)
	void createLabel();

	void ROITrigger();
	void ROIDecision(QAbstractButton *sender);
	void ROISelection(const QRect &roi);

	void openContextMenu();

	void screenshot();

	void updateRGB(bool success);

signals:
	void clearLabel();
	void alterLabel(const multi_img::Mask &mask, bool negative);
	void newLabelColors(const QVector<QColor> &colors, bool changed);
	void drawOverlay(const multi_img::Mask &mask);
	void seedingDone(bool yeah = false);

protected:
	enum normMode {
		NORM_OBSERVED = 0,
		NORM_THEORETICAL = 1,
		NORM_FIXED = 2
	};

	class RgbSerial : public MultiImg::BgrSerial {
	public:
		RgbSerial(multi_img_ptr multi, mat_vec3f_ptr bgr, qimage_ptr rgb) 
			: MultiImg::BgrSerial(multi, bgr), rgb(rgb) {}
		virtual ~RgbSerial() {};
		virtual bool run();
	protected:
		qimage_ptr rgb;
	};

	class RgbTbb : public MultiImg::BgrTbb {
	public:
		RgbTbb(multi_img_ptr multi, mat_vec3f_ptr bgr, qimage_ptr rgb) 
			: MultiImg::BgrTbb(multi, bgr), rgb(rgb) {}
		virtual ~RgbTbb() {}
		virtual bool run();
	protected:
		class Rgb {
		public:
			Rgb(cv::Mat_<cv::Vec3f> &bgr, QImage &rgb) 
				: bgr(bgr), rgb(rgb) {}
			void operator()(const tbb::blocked_range2d<int> &r) const;
		private:
			cv::Mat_<cv::Vec3f> &bgr;
			QImage &rgb;
		};

		qimage_ptr rgb;
	};

	class NormRangeTbb : public MultiImg::DataRangeTbb {
	public:
		NormRangeTbb(multi_img_ptr multi, 
			data_range_ptr range, normMode mode, int target, 
			multi_img::Value minval, multi_img::Value maxval, bool update,
			cv::Rect targetRoi = cv::Rect(0, 0, 0, 0)) 
			: MultiImg::DataRangeTbb(multi, range, targetRoi), 
			mode(mode), target(target), minval(minval), maxval(maxval), update(update) {}
		virtual ~NormRangeTbb() {}
		virtual bool run();
	protected:
		normMode mode;
		int target;
		multi_img::Value minval;
		multi_img::Value maxval;
		bool update;
	};

    void changeEvent(QEvent *e);

	/* helper functions */
	void labelflush();
	void applyROI(bool reuse);
	void labelmask(bool negative);
	// returns true if updates were triggered, false if not (trigger yourself!)
	bool setLabelColors(const std::vector<cv::Vec3b> &colors);
	void setLabels(const vole::Labeling &labeling);

	void runGraphseg(const multi_img& input, const vole::GraphSegConfig &config);

	// multispectral image and gradient
	multi_img_ptr full_image, image, gradient, imagepca, gradientpca;
	// current region of interest
	cv::Rect roi;
	// bands from all representations (image, gradient, PCA..)
	std::vector<std::vector<QPixmap*> > bands;
	// label colors
	QVector<QColor> labelColors;
	// full image labels and roi scoped labels
	cv::Mat1s full_labels, labels;

	// rgb pixmap
	QPixmap full_rgb, rgb;
	qimage_ptr full_rgb_temp; // QPixmap cannot be directly shared between threads

	// viewers
	std::vector<multi_img_viewer*> viewers;
	multi_img_viewer *activeViewer;

	normMode normIMG, normGRAD;
	data_range_ptr normIMGRange, normGRADRange;

private:
	void initUI();
	void initGraphsegUI();
	void initIlluminantUI();
#ifdef WITH_SEG_MEANSHIFT
	void initUnsupervisedSegUI();
#endif
	void initNormalizationUI();
	void updateBand();
	void buildIlluminant(int temp);

	// cache for illumination coefficients
	typedef std::map<int, std::pair<
			Illuminant, std::vector<multi_img::Value> > > Illum_map;
	Illum_map illuminants;

	CommandRunner *usRunner;

	QMenu *contextMenu;
};

#endif // VIEWERWINDOW_H
