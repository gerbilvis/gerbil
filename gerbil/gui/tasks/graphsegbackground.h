#ifndef GRAPHSEGBACKGROUND_H
#define GRAPHSEGBACKGROUND_H

#include <graphseg.h>

class GraphsegBackground : public BackgroundTask {
public:
	GraphsegBackground(const vole::GraphSegConfig &config, SharedMultiImgPtr input,
		const cv::Mat1s &seedMap, boost::shared_ptr<cv::Mat1s> result)
		: config(config), input(input), seedMap(seedMap), result(result) {}
	virtual ~GraphsegBackground() {}
	virtual bool run()	{
		vole::GraphSeg seg(config);
		*(result.get()) = seg.execute(**input, seedMap);
		return true;
	}

protected:
	vole::GraphSegConfig config;
	SharedMultiImgPtr input;
	cv::Mat1s seedMap;
	boost::shared_ptr<cv::Mat1s> result;
};

#endif // GRAPHSEGBACKGROUND_H
