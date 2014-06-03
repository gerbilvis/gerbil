#ifndef GRAPH_SEG_TASK_H
#define GRAPH_SEG_TASK_H

#include <graphseg.h>

class GraphSegTask : public BackgroundTask {
public:
	GraphSegTask(const seg_graphs::GraphSegConfig &config,
				 SharedMultiImgPtr input,
				 const cv::Mat1s &seedMap, boost::shared_ptr<cv::Mat1s> result)
		: config(config), input(input), seedMap(seedMap), result(result) {}
	virtual ~GraphSegTask() {}
	virtual bool run()	{
		seg_graphs::GraphSeg seg(config);
		*(result.get()) = seg.execute(**input, seedMap);
		return true;
	}

protected:
	seg_graphs::GraphSegConfig config;
	SharedMultiImgPtr input;
	cv::Mat1s seedMap;
	boost::shared_ptr<cv::Mat1s> result;
};

#endif // GRAPH_SEG_TASK_H
