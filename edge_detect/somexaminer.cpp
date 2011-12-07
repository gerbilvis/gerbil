#include "somexaminer.h"

#ifdef WITH_GERBIL_COMMON
#include "illuminant.h"
#endif

#include <cv.h>
#include <highgui.h>

/* mouse handler indicators */
bool mouse_left_clicked;
bool mouse_right_clicked;
unsigned int g_node1=UINT_MAX; // TODO WTF
unsigned int g_node2=UINT_MAX;

/// global mouse handler function controlling the leftclick in the msi window
void mouse_leftclick(int event, int x, int y, int flags, void* param) {

  cv::Point *mark = (cv::Point *) param;

  switch (event) {
  case CV_EVENT_MOUSEMOVE:
    mark->x = x;
    mark->y = y;
    break;

  case CV_EVENT_LBUTTONDOWN:
    mouse_left_clicked = true;
    break;

  case CV_EVENT_LBUTTONUP:
    mouse_left_clicked = false;
    break;
  }
}

/// global mouse handler function controlling the rightclick in the som window
void mouse_rightclick(int event, int x, int y, int flags, void* param) {

  cv::Point *mark2 = (cv::Point *) param;

  switch( event ){
    case CV_EVENT_MOUSEMOVE:
      mark2->x = x;
      mark2->y = y;
      break;

    case CV_EVENT_RBUTTONDOWN:
      mouse_right_clicked = true;
      break;

    case CV_EVENT_RBUTTONUP:
      mouse_right_clicked = false;
      break;

    case CV_EVENT_LBUTTONDOWN:
      mouse_left_clicked = true;
      break;

    case CV_EVENT_LBUTTONUP:
      mouse_left_clicked = false;
      break;
    }
}

SOMExaminer::SOMExaminer()
{
	// just cut out ".txt" from the name ...
    if (m_msiName.find(".txt")) {
      m_msiName.resize(m_msiName.size() - 4);
    }
}

void SOMExaminer::displayGraphDistances()
{

	displaySom(true);

  mouse_left_clicked = false;

  cv::Point mark(0,0);
  cv::Point mark2(0,0);

  double scaleX = (double)(1.0*m_msi.height) / (m_widthSom);
  double scaleY = (double)(1.0*m_msi.height) / (m_heightSom);

	cv::Mat3f som;
  cv::Mat3f somCopy;

	cv::resize(m_img_som, som, cv::Size(256,256), 3.0, 3.0, cv::INTER_NEAREST);

  som.copyTo(somCopy);

  cv::namedWindow("SOM", CV_GUI_EXPANDED);

  cvSetMouseCallback( "SOM", mouse_rightclick, (void*)&mark2);

	cv::imshow("SOM", somCopy);

	bool display = true;

	while(display)
	{

		if(mouse_right_clicked)
		{
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom)
			{
				std::stringstream ss;
				unsigned int index = (somPoint.y * m_som.getWidth() + somPoint.x);
				g_node2 = index;
				const msi::Node* node = msi_graph->getNode(index);
				const std::vector<msi::Node*>& edg = node->getEdges();

				ss << "(" << somPoint.y << "," << somPoint.x << ") index: " << index;

				for(unsigned int n = 0; n < edg.size(); n++)
				{
					int col, row;
					col =  (edg.at(n))->getIndex() % m_som.getWidth();
					row =  (edg.at(n))->getIndex() / m_som.getWidth();
					cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
					cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

// 					if(edg.size()== 11)cv::rectangle( som, start, end, cv::Scalar(0,255,255), 2 );
// 					if(edg.size()== 10)cv::rectangle( som, start, end, cv::Scalar(0,255,196), 2 );
// 					if(edg.size()== 9)cv::rectangle( som, start, end, cv::Scalar(0,255,96), 2 );
// 					if(edg.size()== 8)cv::rectangle( som, start, end, cv::Scalar(0,255,0), 2 );
// 					if(edg.size()== 7)cv::rectangle( som, start, end, cv::Scalar(255,196,0), 2 );
// 					if(edg.size()== 6)cv::rectangle( som, start, end, cv::Scalar(255,155,0), 2 );
// 					if(edg.size()== 5)cv::rectangle( som, start, end, cv::Scalar(255,128,0), 2 );
// 					if(edg.size()== 4)cv::rectangle( som, start, end, cv::Scalar(255,108,0), 2 );
// 					if(edg.size()== 3)cv::rectangle( som, start, end, cv::Scalar(255,78,0), 2 );
// 					if(edg.size()== 2)cv::rectangle( som, start, end, cv::Scalar(255,55,0), 2 );
// 					if(edg.size()== 1)cv::rectangle( som, start, end, cv::Scalar(255,28,0), 2 );

				}

				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				std::cout << start.x << " " << start.y << "   " <<end.x << " " <<end.y << std::endl;
				cv::rectangle( som, start, end, cv::Scalar(0,0,255), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);

				cv::imshow("SOM", som);
				som = somCopy.clone();

				if(g_node1 != UINT_MAX && g_node2 != UINT_MAX)
				{
					std::cout << "Distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,false,true) <<std::endl;
					if(m_withUMap)
						std::cout << "Weighted distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,true,true)<<std::endl;;
					std::vector<unsigned int> path = msi_graph->getPath(g_node1,g_node2);

					for(unsigned int n = 0; n < path.size();n++)
					{
						int col, row;
						col =  path.at(n) % m_som.getWidth();
						row =  path.at(n) / m_som.getWidth();
						cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
						cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

						cv::rectangle( som, start, end, cv::Scalar(255,255,0), 2 );
					}
					if(m_withUMap)
					{
						std::vector<unsigned int> weightedPath = msi_graph->getWeightedPath(g_node1,g_node2);
						for(unsigned int n = 0; n < weightedPath.size();n++)
						{
							int col, row;
							col =  weightedPath.at(n) % m_som.getWidth();
							row =  weightedPath.at(n) / m_som.getWidth();
							cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
							cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

							cv::rectangle( som, start, end, cv::Scalar(255,0,255), 2 );
						}
					}

					std::cout <<std::endl;
				}
			}
		}
		if(mouse_left_clicked)
		{
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom)
			{
				std::stringstream ss;
				unsigned int index = (somPoint.y * m_som.getWidth() + somPoint.x);
				g_node1 = index;
				const msi::Node* node = msi_graph->getNode(index);
				const std::vector<msi::Node*>& edg = node->getEdges();

				ss << "(" << somPoint.y << "," << somPoint.x << ") index: " << index;

				for(unsigned int n = 0; n < edg.size(); n++)
				{
					int col, row;
					col =  (edg.at(n))->getIndex() % m_som.getWidth();
					row =  (edg.at(n))->getIndex() / m_som.getWidth();
					cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
					cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

// 					if(edg.size()== 11)cv::rectangle( som, start, end, cv::Scalar(0,255,255), 2 );
// 					if(edg.size()== 10)cv::rectangle( som, start, end, cv::Scalar(0,255,196), 2 );
// 					if(edg.size()== 9)cv::rectangle( som, start, end, cv::Scalar(0,255,96), 2 );
// 					if(edg.size()== 8)cv::rectangle( som, start, end, cv::Scalar(0,255,0), 2 );
// 					if(edg.size()== 7)cv::rectangle( som, start, end, cv::Scalar(255,196,0), 2 );
// 					if(edg.size()== 6)cv::rectangle( som, start, end, cv::Scalar(255,155,0), 2 );
// 					if(edg.size()== 5)cv::rectangle( som, start, end, cv::Scalar(255,128,0), 2 );
// 					if(edg.size()== 4)cv::rectangle( som, start, end, cv::Scalar(255,108,0), 2 );
// 					if(edg.size()== 3)cv::rectangle( som, start, end, cv::Scalar(255,78,0), 2 );
// 					if(edg.size()== 2)cv::rectangle( som, start, end, cv::Scalar(255,55,0), 2 );
// 					if(edg.size()== 1)cv::rectangle( som, start, end, cv::Scalar(255,28,0), 2 );
				}

				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				cv::rectangle( som, start, end, cv::Scalar(0,0,255), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);

				cv::imshow("SOM", som);
				som = somCopy.clone();

				if(g_node1 != UINT_MAX && g_node2 != UINT_MAX)
				{
					std::cout << "Distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,false,true)<<std::endl;;
					if(m_withUMap)
						std::cout << "Weighted distance " << g_node1 << " -- " << g_node2 << " : " << msi_graph->getDistance(g_node1,g_node2,true,true)<<std::endl;;

					if(!msi_graph->periodic())
					{
						std::vector<unsigned int> path = msi_graph->getPath(g_node1,g_node2);
						for(unsigned int n = 0; n < path.size();n++)
						{
							int col, row;
							col =  path.at(n) % m_som.getWidth();
							row =  path.at(n) / m_som.getWidth();
							cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
							cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

							cv::rectangle( som, start, end, cv::Scalar(255,255,0), 2 );
						}
						if(m_withUMap)
						{
							std::vector<unsigned int> weightedPath = msi_graph->getWeightedPath(g_node1,g_node2);
							for(unsigned int n = 0; n < weightedPath.size();n++)
							{
								int col, row;
								col =  weightedPath.at(n) % m_som.getWidth();
								row =  weightedPath.at(n) / m_som.getWidth();
								cv::Point start(static_cast<int>(col * scaleX), static_cast<int>(row * scaleY));
								cv::Point end(static_cast<int>((col + 1) * scaleX), static_cast<int>((row + 1) * scaleY));

								cv::rectangle( som, start, end, cv::Scalar(255,0,255), 2 );
							}
						}
					}
				}
			}
		}

		cv::waitKey(60);
	}
}

void SOMExaminer::compareMultispectralData()
{
#ifdef WITH_GERBIL_COMMON
	displayMSI(true);
	displaySom(true);

	mouse_left_clicked = false;

	cv::Point mark(0,0);
	cv::Point mark2(0,0);

	double scaleX = (double)(m_msi.width) / (m_widthSom);
	double scaleY = (double)(m_msi.height) / (m_heightSom);

	cv::Mat msi;
	cv::Mat3f som;
	cv::Mat3f somCopy;
	cv::resize(m_img_som, som, cv::Size(m_msi.width, m_msi.height), 0., 0., cv::INTER_NEAREST);
	som.copyTo(somCopy);

	m_img_msi = m_msi.bgr();
	m_img_msi.convertTo( msi, CV_8UC3, 255.);


	cv::namedWindow("Multispectral Image", 1);
	cv::namedWindow("SOM", 1);

	cvSetMouseCallback( "Multispectral Image", mouse_leftclick, (void*)&mark);
	cvSetMouseCallback( "SOM", mouse_rightclick, (void*)&mark2);

	cv::imshow("Multispectral Image", msi);
	cv::imshow("SOM", som);

  if( !m_img_rank.empty() )
	{
		cv::imshow("Rank Image", m_img_rank);
  }

	int rank = 0;
	int posX = mark.x;
	int posY = mark.y;

	while(true) {

		if(mouse_right_clicked) {
			cv::Point somPoint(static_cast<int>(mark2.x/scaleX) , static_cast<int>(mark2.y/scaleY));
			if(somPoint.x < m_widthSom && somPoint.y < m_heightSom) {

				if (!m_rank.empty())
					 rank = m_rank.at<int>(somPoint);
				else
					rank = somPoint.x;

				std::stringstream ss;
				ss << rank;

				cv::Point start(static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
				cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y + 1) * scaleY));
				cv::rectangle( som, start, end, cv::Scalar(0,0,0), 2 );
				cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y + 0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
				std::cout << rank << " " << std::endl;

				cv::imshow("SOM", som);
				som = somCopy.clone();
			}
		}

		if (mouse_left_clicked && mark.y < m_msi.height && mark.x < m_msi.width) {
			const multi_img::Pixel& atClick = m_msi(mark.y, mark.x);
			const multi_img::Pixel& old_click = m_msi(posY,posX);
			std::cout << " Difference to previous clicked vector: "
					<< m_som.distfun->getSimilarity(old_click, atClick) << std::endl;
			posX = mark.x;
			posY = mark.y;
			cv::Point somPoint = m_som.identifyWinnerNeuron(atClick);

			if( !m_rank.empty() )
				rank = m_rank.at<int>(somPoint);
      		else
				rank = somPoint.x;

			std::stringstream ss;
			ss << rank;

			cv::Point start( static_cast<int>(somPoint.x * scaleX), static_cast<int>(somPoint.y * scaleY));
			cv::Point end(static_cast<int>((somPoint.x + 1) * scaleX), static_cast<int>((somPoint.y +1) * scaleY));
			cv::rectangle( som, start, end, cv::Scalar(0,0,0), 2 );
			cv::putText( som, ss.str(), cv::Point(static_cast<int>((somPoint.x + 0.2)*scaleX),static_cast<int>((somPoint.y+0.5)*scaleY)), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(1., 1., 1.), 2, 8, false);
			std::cout << "Rank: " << rank << " Difference to input: "
				<< m_som.distfun->getSimilarity(atClick, *m_som.getNeuron(somPoint.x, somPoint.y))
				<< std::endl;

			cv::imshow("SOM", som);
			som = somCopy.clone();
		}
    cv::waitKey(60);
  }
#endif
}

void SOMExaminer::displaySom(bool write)
{
#ifdef WITH_GERBIL_COMMON
  /* NOTE: if we do not have the source image, we do not know _anything_ about
     its filters. So we _cannot_ create a truthful RGB image.
     If som needs to be displayed without image at hand, store the meta data
     and create empty image with correct meta data.
   */
	if (m_img_som.empty())
	{
		m_img_som = cv::Mat3f(m_som.getHeight(), m_som.getWidth());
	}

	for(int y = 0; y < m_heightSom; y++) {
		cv::Vec3f *row = m_img_som[y];
		for(int x = 0; x < m_widthSom; x++) {
			Neuron *n = m_som.getNeuron(x,y);

			multi_img::Pixel p(*n);
			row[x] = m_msi.bgr(p);
		}
	}

	cv::Mat3f somShow;
	int height;
	if (m_heightSom == 1)
	{
		height = 100;
		cv::resize(m_img_som, somShow, cv::Size(m_msi.width, height), 0., 0., cv::INTER_NEAREST);
	}

	if (write)
	{
		m_img_som.convertTo(somShow, CV_8UC3, 255.);
		std::string graph = "";

		if(m_withUMap)
			graph += "";

    cv::imwrite(config.output_dir+getFilenameExtension()+ graph +"_som.png", somShow);
	std::cout << "wrote SOM to " << config.output_dir+getFilenameExtension()+ graph +"_som.png" << std::endl;
  }
#endif
}

void SOMExaminer::writeSom(int iter)
{
#ifdef WITH_GERBIL_COMMON
	cv::Mat somShow;
	int height;

	if(m_currIter != iter)
		return;

	if (m_img_som.empty())
	{
		m_img_som = cv::Mat3f(m_som.getHeight(), m_som.getWidth());
	}

	for(int y = 0; y < m_heightSom; y++)
	{
		cv::Vec3f *row = m_img_som[y];
		for(int x = 0; x < m_widthSom; x++)
		{
			Neuron *n = m_som.getNeuron(x,y);
			multi_img::Pixel p(*n);
			row[x] = m_msi.bgr(p);
		}
	}

	if (m_heightSom == 1)
	{
		cv::Mat somScaled;
		height = 100;
		cv::resize(m_img_som, somScaled, cv::Size(m_msi.width, height), 0., 0., cv::INTER_NEAREST);
		somScaled.convertTo(somShow, CV_8UC3, 255.);
	}
	else
		m_img_som.convertTo(somShow, CV_8UC3, 255.);

	std::stringstream iterStream;
	iterStream << iter;
	std::string graph = "";

  cv::imwrite(config.output_dir+getFilenameExtension()+ graph +"_som_" + iterStream.str() + ".png", somShow);
#endif
}

void SOMExaminer::displayGraph(bool write) {

#ifdef WITH_GERBIL_COMMON
	assert(msi_graph);

	cv::Mat3f graphSom = cv::Mat3f(m_heightSom,m_widthSom);

	for(int y = 0; y < m_heightSom; y++)
	{
		cv::Vec3f *vec = graphSom[y];
		for(int x = 0; x < m_widthSom; x++)
		{
			vec[x] = msi_graph->getNode(y* m_widthSom + x)->getNeuron()->getRGB();
		}
	}

	cv::Mat3f graphSomShow;

	int height;
	if (m_heightSom == 1)
		height = 100;
	else if (m_msi.empty())
		height = 512;
	else
		height = m_msi.width;

	if (m_msi.empty())
		cv::resize(graphSom, graphSomShow, cv::Size(512, height), 0., 0.,
                 cv::INTER_NEAREST);
	else
		cv::resize(graphSom, graphSomShow, cv::Size(m_msi.width, height), 0., 0.,
                 cv::INTER_NEAREST);


	if (write)
	{
		graphSomShow.convertTo(graphSomShow, CV_8UC3, 255.);
		std::string graph = "";
		cv::imwrite(config.output_dir+ getFilenameExtension() + graph + "_graphSOM.png", graphSomShow);
	}
#endif
}

void SOMExaminer::displayBmuMap(bool write)
{
	cv::Mat_<uchar> bmuShow(m_heightSom, m_widthSom);
	double maxIntensity = 0.0;

	for(int y = 0; y < m_bmuMap.rows; y++)
	{
		double *rowPtr =  m_bmuMap[y];
		for(int x = 0; x < m_bmuMap.cols; x++)
		{
			//lift values logarithmic for viewing purpose
			rowPtr[x] += 1.0;
			rowPtr[x] = std::log((rowPtr[x]));

			if(rowPtr[x] > maxIntensity)
			{
				maxIntensity = rowPtr[x];
			}
		}
	}

	for(int y = 0; y < m_bmuMap.rows; y++)
	{
		double *rowPtr =  m_bmuMap[y];
		uchar * showPtr = bmuShow.ptr<uchar>(y);
		for(int x = 0; x < m_bmuMap.cols; x++)
		{
			showPtr[x] = static_cast<uchar>( 255.0 * (rowPtr[x] / maxIntensity));
		}
	}

	if (write)
	{
		cv::imwrite(config.output_dir+ getFilenameExtension() + "_bmu_map.png", bmuShow);
		//write gnuplot data
		std::ofstream out;
		std::string fn = config.output_dir+getFilenameExtension() +"_bmu_map.dat";
		out.open(fn.c_str(), std::ofstream::out | std::ofstream::trunc);
		assert(out.is_open());

		out << "# Number of winnings for neurons "<< std::endl;
		out << "# x\ty\tWinnings\t"<< std::endl;

		for(int y = 0; y < m_bmuMap.rows ; y++ )
		{
			double *uPtr = m_bmuMap[y];

			for(int x = 0; x < m_bmuMap.cols; x++ )
			{
				out << x <<"\t" << y <<  "\t" <<uPtr[x]<<"\t" << std::endl;
			}
			out << "\n";
		}
		out.close();
  }
}
