//	-------------------------------------------------------------------------------------------------------------	 	//
// 														Variants of Self-Organizing Maps																											//
// Studienarbeit in Computer Vision at the Chair of Patter Recognition Friedrich-Alexander Universitaet Erlangen		//
// Start:	15.11.2010																																																//
// End	:	16.05.2011																																																//
// 																																																									//
// Ralph Muessig																																																		//
// ralph.muessig@e-technik.stud.uni-erlangen.de																																			//
// Informations- und Kommunikationstechnik																																					//
//	---------------------------------------------------------------------------------------------------------------	//


#include <iostream>
#include "fastDijkstra.h"

#define WHITE 0
#define BLACK 1

using namespace std;

fastDijkstra::fastDijkstra(unsigned int nnodes)
{
	unsigned int size = nnodes;
	m_distanceCache = cv::Mat::zeros(size,size, CV_64FC2);
	for(unsigned int y = 0; y < size; y++)
	{	
		cv::Vec2d *distPtr = m_distanceCache[y];
		for(unsigned int x= 0; x < size; x++)
		{	
			distPtr[x][0] = DBL_MAX;
			distPtr[x][1] = DBL_MAX;
		}
	}
	
	m_nodes_number = nnodes;
	adj = new list<Node*>[m_nodes_number];
	colour = new unsigned int[m_nodes_number];
	dist = new double[m_nodes_number];
	pie = new unsigned int[m_nodes_number];
}

void fastDijkstra::showCache()
{
	for(int y = 0; y < m_distanceCache.rows; y++)
	{
		cv::Vec2d *wPtr = m_distanceCache[y];
		for(int x = 0; x < m_distanceCache.rows; x++)
		{
			std::cout << wPtr[x][0] << " ";
		}
		std::cout << std::endl;
	}
}

fastDijkstra::~fastDijkstra()
{

	delete[] colour;
	delete[] dist;
	delete[] pie;
	adj->clear();
	delete adj;
}


void fastDijkstra::addEdge(unsigned int from,unsigned int to, double edgeWeight)
{
	adj[from].push_back(new Node(to,edgeWeight));
	adj[to].push_back(new Node(from,edgeWeight));
}

double fastDijkstra::getEdgeWeight(unsigned int from,unsigned int to)
{
	std::list<Node*>::iterator iter;
	for(iter = adj[from].begin(); iter != adj[from].end(); iter++ )
	{
		if((*iter)->iVertexNo == to)
			return (*iter)->distance;
	}	
	return DBL_MAX;
}

void fastDijkstra::removeEdge(unsigned int from,unsigned int to)
{
	std::list<Node*>::iterator iter;
	for(iter = adj[from].begin(); iter != adj[from].end(); iter++ )
	{
		if((*iter)->iVertexNo == to)
		{
			adj->erase(iter);
			m_distanceCache[from][to] = DBL_MAX;

			break;
		}	
	}	
	
	for(iter = adj[to].begin(); iter != adj[to].end(); iter++ )
	{
		if((*iter)->iVertexNo == from)
		{
			adj->erase(iter);
			m_distanceCache[to][from] = DBL_MAX;

			break;
		}	
	}	
}

void fastDijkstra::scaleDistances(double scaling)
{
	assert(scaling >=0.0);
		
	list<Node*>::iterator it;

	for(unsigned int i = 0; i < m_nodes_number; i++)
	{	
		for ( it=adj[i].begin() ; it != adj[i].end(); it++ )
		{
			(*it)->weightedDistance =  (1.0 -scaling) *((*it)->distance ) + 10.0 *scaleWeight( (*it)->edgeWeight,scaling);
		}	
	}
	
}

double fastDijkstra::scaleWeight(double weight, double scale)
{	
	assert(scale >= 0.0);
	
	double res =   scale * (weight);
	return  res; 
}

void fastDijkstra::setEdgeWeight(unsigned int from,unsigned int to, double weight)
{
	list<Node*>::iterator it;
	for ( it=adj[from].begin() ; it != adj[from].end(); it++ )
	{
		if((*it)->iVertexNo == to)
		{	
			(*it)->edgeWeight = weight/ (*it)->distance; //diagonal edges get weighted by sqrt(2)

			break;
		}	
		
	}
	for ( it=adj[to].begin() ; it != adj[to].end(); it++ )
	{
		if((*it)->iVertexNo == from)
		{	
			(*it)->edgeWeight = weight/ (*it)->distance; //diagonal edges get weighted by sqrt(2)
			break;
		}	
		
	}	
	

}

void fastDijkstra::dumpArrays()
{
	for(unsigned int i = 0; i< m_nodes_number;i++)
	{
		std::cout << "Node: "<<i << " "<< pie[i] << " " << dist[i] <<std::endl;
	}	
}

void fastDijkstra::printPath(unsigned int s, unsigned int d)
{

	if((d == s))
	{	
		std::cout << " " << s;
			return;
	}	
	else
	{	
		printPath(s, pie[d]);
		cout << "---->" << d;
	}
	std::cout << std::endl;
	return;
}

void fastDijkstra::path(unsigned int s, unsigned int d, std::vector<unsigned int> *v)
{
	if (d == 0)
	{
		return;//TODO warum
	}	

	if(v->size() > 100U)
	{
		for(unsigned int i = 0; i < 100U;i++)
		std::cout << "CRITICAL: " << s << " -- " << d << " v size: " << v->at(i) << std::endl;
	}

	if((d == s))
	{	
		return ;
	}	
	else
	{	
		v->push_back(d);
		path(s, pie[d],v);
	}
	return;
}


double fastDijkstra::allPath(unsigned int source, unsigned int destination, bool weighted)
{
	double dij,z;	
	unsigned int markedNodes = 0;
	unsigned int p = 0;
	list<Node*>::iterator it;

	for (unsigned int i1=0; i1<m_nodes_number; i1++)
	{
		dist[i1] = DBL_MAX;
		colour[i1] = WHITE;
	}
	dist[source] = 0;
	unsigned int i = source; // Latest vertex permanently labeled
	colour[source] = BLACK;
	pie[source] = 0;
	while (markedNodes != m_nodes_number)
	{
		double m = DBL_MAX; // smallest distance
		for (unsigned int j=0; j<m_nodes_number; j++)
		{

			if (colour[j]==BLACK) continue;
			else
			{
				it = adj[i].begin();
				
				while (it != adj[i].end())
				{
					if ((*it)->iVertexNo == j) break;
					else it++;
				}				
				if (it==adj[i].end()) 
					dij = DBL_MAX;
				else
				{
					if(weighted)
						dij = (*it)->weightedDistance;
					else
						dij = (*it)->distance;
				}

				if((dij == DBL_MAX || dist[i] == DBL_MAX))
					z = DBL_MAX;
				else
					z = dij + dist[i];

				if (z < dist[j])
				{
					dist[j] = z;
					pie[j] = i;
				}				
				if (dist[j] < m)
				{
					m = dist[j];

					p = j;					
				}
			}
		}
		colour[p] = BLACK;
		markedNodes++;
		i = p;
	}
	return dist[destination];

}


double fastDijkstra::shortestPath(unsigned int source, unsigned int destination, bool weighted)
{
	
	if(source == destination)
	{
		dist[source] = 0.0;
		return 0.0;
	}	
	double dij,z;	
	unsigned int p = 0;
	list<Node*>::iterator it;
	for (unsigned int i1=0; i1<m_nodes_number; i1++)
	{
		dist[i1] = DBL_MAX;
		colour[i1] = WHITE;
	}
	dist[source] = 0;
	unsigned int i = source; // Latest vertex permanently labeled
	colour[source] = BLACK;
	pie[source] = 0;
	while (p != destination)
	{
		double m = DBL_MAX; // smallest distance
		for (unsigned int j=0; j<m_nodes_number; j++)
		{

			if (colour[j]==BLACK) continue;
			else
			{
				it = adj[i].begin();
				
				while (it != adj[i].end())
				{
					if ((*it)->iVertexNo == j) break;
					else it++;
				}				
				if (it==adj[i].end()) 
					dij = DBL_MAX;
				else
				{
					if(weighted)
						dij = (*it)->weightedDistance;
					else
						dij = (*it)->distance;
				}
				if((dij == DBL_MAX || dist[i] == DBL_MAX))
					z = DBL_MAX;
				else
					z = dij + dist[i];

				if (z < dist[j])
				{
					dist[j] = z;
					pie[j] = i;
				}				
				if (dist[j] < m)
				{
					m = dist[j];

					p = j;
				}
			}
		}
		colour[p] = BLACK;
		i = p;
	}
	return dist[destination];

}

double fastDijkstra::getDiameter()
{
	double diameter = 0.0;
	
	
	for(int y = 0; y < m_distanceCache.rows; y++)
	{
		cv::Vec2d *cachePtr = m_distanceCache[y];
		for(int x = 0; x < m_distanceCache.rows; x++)
		{
			if(cachePtr[x][0] != DBL_MAX && cachePtr[x][0] >diameter)
				diameter = cachePtr[x][0];
		}	
	}

	return diameter;
}

double fastDijkstra::characteristicPathLength()
{
	double length = 0.0;
	std::list<double> means;
	
	for(int y = 0; y < m_distanceCache.rows; y++)
	{
		cv::Vec2d *cachePtr = m_distanceCache[y];
		double m = 0.0;
		for(int x = 0; x < m_distanceCache.rows; x++)
		{
			m += cachePtr[x][0];
		}	
		m /= (double)(m_distanceCache.rows -1);
		means.push_back(m);
		
	}
	
	//distances are ordered ascending
	means.sort();
	
	std::list<double>::iterator it = means.begin();
	// pick median
	if(means.size() % 2 == 1)
	{
		std::advance(it, means.size()/2);
		length = *it;
	}
	else if(means.size() % 2 == 0)
	{
		std::advance(it, means.size()/2);
		length = *it;
		it--;
		length += *it;
		length /= 2.0;
	}	

	return length;
}



std::vector<unsigned int> fastDijkstra::getPath(unsigned int indexA, unsigned int indexB)
{
	shortestPath(indexA,indexB);
	std::vector<unsigned int> v;
	path(indexA,indexB, &v);
	
	return v;
}

std::vector<unsigned int> fastDijkstra::getWeightedPath(unsigned int indexA, unsigned int indexB)
{
	shortestPath(indexA,indexB,true);
	std::vector<unsigned int> v;
	path(indexA,indexB, &v);

	return v;
}

double fastDijkstra::getDistance(unsigned int indexA, unsigned int indexB,bool getWeightedDistance, bool showPath)
{
	unsigned int indexMin = std::min(indexA,indexB);
	unsigned int indexMax = std::max(indexA,indexB);
	
	if(showPath)
	{
		double distance;
		distance = shortestPath(indexMin,indexMax, getWeightedDistance);
		printPath(indexMin,indexMax);
		return distance;
	}	
		
	
	cv::Vec2d *distPtr = m_distanceCache[indexMin];
	if(getWeightedDistance)
	{	
		if(distPtr[indexMax][1] != DBL_MAX)
		{
			return distPtr[indexMax][1];
		}
	}
	else
	{
		if(distPtr[indexMax][0] != DBL_MAX)
		{
			return distPtr[indexMax][0];
		}	
	}

	
	distPtr = m_distanceCache[indexMax];
	if(getWeightedDistance)
	{	
		if(distPtr[indexMin][1] != DBL_MAX)
		{
			return distPtr[indexMin][1];
		}
	}
	else
	{
		if(distPtr[indexMin][0] != DBL_MAX)
		{
			return distPtr[indexMin][0];
		}	
	}
	
	// no cached distance available
	double distance;
	double val = DBL_MAX;

	distance = allPath(indexMin, indexMax, getWeightedDistance);
	distPtr = m_distanceCache[indexMin];
	for(int x = 0; x < m_distanceCache.cols;x++)
	{
		if(getWeightedDistance)
		{
			if(distPtr[x][1] == DBL_MAX)
			{	
				val = dist[x];
				
				if(val != DBL_MAX)
				{
					distPtr[x][1] = val;
				}	
			}		
		}	
		else
		{
			if(distPtr[x][0] == DBL_MAX)
			{	
				val = dist[x];
				
				if(val != DBL_MAX)
				{
					distPtr[x][0] = val;
				}	
			}	
		}	
	}

	return distance;
}
