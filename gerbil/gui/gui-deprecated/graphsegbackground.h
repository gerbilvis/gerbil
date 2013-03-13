#ifndef GRAPHSEGBACKGROUND_H
#define GRAPHSEGBACKGROUND_H

class GraphsegBackground : public BackgroundTask {
public:
	GraphsegBackground(const vole::GraphSegConfig &config, multi_img_ptr input,
		const cv::Mat1s &seedMap, boost::shared_ptr<multi_img::Mask> result)
		: config(config), input(input), seedMap(seedMap), result(result) {}
	virtual ~GraphsegBackground() {}
	virtual bool run()	{
		vole::GraphSeg seg(config);
		*(result.get()) = seg.execute(**input, seedMap);
		return true;
	}

protected:
	vole::GraphSegConfig config;
	multi_img_ptr input;
	cv::Mat1s seedMap;
	boost::shared_ptr<multi_img::Mask> result;
};

#endif // GRAPHSEGBACKGROUND_H
