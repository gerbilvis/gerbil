#ifndef SEG_FELZENSZWALB_CONFIG_H
#define SEG_FELZENSZWALB_CONFIG_H

#include "vole_config.h"

#include <vector>

#ifdef WITH_QT
#include <QWidget>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLineEdit>
#include <QCheckBox>
#endif // WITH_QT

namespace vole {
 /** \addtogroup VoleModules */

/** Configuration parameters for the Felzenszwalb/Huttenlocher segmentation
 * \ingroup VoleModules
 */
class SegFelzenszwalbConfig : public Config {
public:
	SegFelzenszwalbConfig(const std::string& prefix = std::string());

	// global vole stuff
	/// may iebv create graphical output (i.e. open graphical windows)?
	bool isGraphical;
	/// input file name
	std::string input_file;
	/// directory for all intermediate files
	std::string output_directory;

	// parameters for the pre-segmentation (felzenszwalb/huttenlocher)
	// sigma for the Felzenszwalb/Huttenlocher segmentation
	double sigma;
	// k for the Felzenszwalb/Huttenlocher segmentation
	int k_threshold;
	// minimum superpixel size for the Felzenszwalb/Huttenlocher segmentation
	int min_size;
	// segment on chromaticity image (instead of intensity image)
	bool chroma_img;

	virtual std::string getString() const;

	virtual ~SegFelzenszwalbConfig();

	#ifdef WITH_QT
	virtual QWidget *getConfigWidget();
	virtual void updateValuesFromWidget();
	QPushButton *getUpdateButton() { return updateComputation; }
	#endif // WITH_QT

protected:
	#ifdef WITH_BOOST
		virtual void initBoostOptions();
	#endif // WITH_BOOST

	#ifdef WITH_QT
	QLineEdit *edit_sigma, *edit_k_threshold, *edit_min_size;
	QCheckBox *chk_chroma_img;
	void initConfigWidget();
	#endif // WITH_QT

	QWidget *configWidget;
	QHBoxLayout *layout;
	QPushButton *updateComputation;

};

}

#endif // SEG_FELZENSZWALB_CONFIG_H
