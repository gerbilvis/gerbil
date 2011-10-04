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


#include <cmath>
#include <time.h>//random number generator

#include <iostream>
#include <fstream>
#include <sstream>
#include "completingGraph.h"

#include "stopwatch.h"

namespace msi {
	CompletingGraph::CompletingGraph(const Configuration &c, unsigned int n, bool directed)
	 : Graph(n, directed), conf(c), intruder(false),
	   start(c.startcomp < 1 ? 1 : c.startcomp), end(c.finishcomp < (start+1) ? start : c.finishcomp - 1)
	{
    
		minimumEdgeCount = getSize();
		maxIter = c.maxIter;
		std::cout << "Maxiter " << maxIter <<std::endl;
		conf = c;
		m_isPeriodic = c.periodic;
	    
	}

	void CompletingGraph::remove(unsigned int index) {
		nodesByGrade.erase(getNode(index));
		Graph::remove(index);
	}

	void CompletingGraph::initCompleter() {
		// fill nodesByGrade
		nodesByGrade.clear();
		for (unsigned int i = 0; i < getSize(); ++i) {
			if (getNode(i) != NULL) {
				nodesByGrade.insert(getNode(i));
			}
		}

		// set edgesPerIter, edgesWantedNow
		int edgeMin = getEdgeCount();
		int edgeMax = getSize() * (getSize()-1);
		if (!conf.directed) {
			edgeMax /= 2;
		}

		edgesPerIter = (double)(edgeMax - edgeMin);
		if (end > start) {
			 edgesPerIter /= (double)(end - start);
		}
		edgesWantedNow = edgeMin;
	}

	void CompletingGraph::proceedCompleter() {
		// calculate how many new edges and call addRandomEdge()
		edgesWantedNow += edgesPerIter;
		for (unsigned int i = getEdgeCount(); i < ceil(edgesWantedNow); ++i) {
			if (!addRandomEdge())
				break;
		}
	}

	void CompletingGraph::restartCompleter() {
		// reset the start and end points to ensure completion is done
		if (start < iter) {
			start = iter;
			if (end < iter)
				end = iter;
			initCompleter();
			proceedCompleter();
		}
	}

	void CompletingGraph::nextIter() {

		Graph::nextIter();
		if (conf.completing) {
			if (iter == start) {
				initCompleter();
			}

			if ((iter == end)||((iter > start)&&(iter <= end))) {
				proceedCompleter();
			}
		}
	}

	bool CompletingGraph::addRandomEdge() {
		const Node* a;
		const Node* b;
		nbgIter i, j;
		/* not really random: search the nodes sorted by degree to
		   add edges on nodes with lesser degree values */
		for (i = nodesByGrade.begin(); i != nodesByGrade.end(); ++i) {
			for (j = i; j != nodesByGrade.end(); ++j) {
				if (j == i) continue;
				/* if we have a directed graph, we have to be able to
				   add edges in both directions. we cannot only test from
				   i to j, but also have to consider adding from j to i */
				bool swap = false;
				if (Node::edgeExists(*i, *j)) {
					if (conf.directed && (!Node::edgeExists(*j, *i)))
						swap = true;
					else continue;
				}
				if (!swap) {
					a = *i; b = *j;
				} else {
					a = *j; b = *i;
				}
				/* we have to re-insert the nodes to get them sorted properly.
				   they have to be removed before the new edge is added, as
				   they wouldn't be found by the multiset anymore otherwise! */
				nodesByGrade.erase(i); nodesByGrade.erase(j);
				addEdge(a->getIndex(), b->getIndex());
				nodesByGrade.insert(a); nodesByGrade.insert(b);
				return true;
			}
		}
		return false;
	}
	
	bool CompletingGraph::removeRandomEdge()
	{
		return false;
	}

	bool CompletingGraph::removeEdge(unsigned int from, unsigned int to)
	{
		if(!edgeExists(from,to))return false;
    
		Graph::removeEdge(from,to);
		
		return true;
	}

	unsigned int CompletingGraph::addMember(Neuron *n) 
	{
		unsigned int newIndex = -1;
		newIndex = Graph::addMember(n);
		if (conf.completing) 
		{
				// we have a new member (_existing_ node!), so we have to
				// include it into the completing process, by restarting it
				restartCompleter();
		}

		intruder = true;
		return newIndex;
	}

	Neuron* CompletingGraph::deleteMember(unsigned int index) 
	{
    Neuron *ret = Graph::deleteMember(index);
		intruder = true;
		if (conf.completing) {
			// restart completing process, to empty cache etc.
			restartCompleter();
		}
		return ret;
	}

	void CompletingGraph::randomizeNodes(unsigned int *field )
	{
		unsigned int nodes = getSize();
		for(unsigned int i =0; i < nodes;i++)
		{  
			do
			{
				field[i]=rand()%nodes;
      
				for(unsigned int i2=0;i2<i;i2++)
				{
					if(field[i]==field[i2])field[i]=UINT_MAX;
				}
			}while(field[i]==UINT_MAX);
		}
	}
	
	void CompletingGraph::createSmallWorld(double p, sw_type type)
	{
		// Deprecated, only useful if other small world substrates besides the mesh (e.g. circular ) are used
	}
  
	double CompletingGraph::calculateEdgeWeight(unsigned int index1, unsigned int index2, int mode)
	{
		assert(index1 < getSize());
		double weight = DBL_MAX;
    
		double sum0 = 0.0;
		double sum1 = 0.0;
		double sum2 = 0.0;
    
		const Node* n1 = getNode(index1);
		Neuron *neu1 = n1->getNeuron(); 
		const std::vector<Node*>& edg = n1->getEdges();
		for(unsigned int i = 0; i < edg.size();i++)
		{
			if( index2 == edg.at(i)->getIndex())
			{
        //default: Euclidean distance
				if( 0 == mode)
				{
					Neuron *neu2 = edg.at(i)->getNeuron();
          
					//make sure every neuron has same length
					assert(neu1->size() == neu2->size());
					multi_img::Value ret = 0.0;
          
					for(unsigned int i = 0; i < neu2->size(); i++) 
					{
						ret += ( ( (*neu1)[i] - (*neu2)[i] ) * ( (*neu1)[i] - (*neu2)[i]) );
					}
          
					//normalize to dimensionality: maximum difference: d * (1.0 - 0.0)
					weight= ret/neu2->size();
					weight = sqrt(weight);
          
					//increase contrast for visulization
					//weight *=10;
					break;
				}
				else if(1 == mode)
				{
					//Spectral angle similarity as presented in
					//"The effectiveness of spectral similarity measures
					//for the analysis of hyperspectral imagery"
					Neuron *neu2 = edg.at(i)->getNeuron();
					assert(neu1->size() == neu2->size());
					multi_img::Value ret = 0.0; 
					sum0 = 0.0;
					sum1 = 0.0;
					sum2 = 0.0;
          
					for(unsigned int i = 0; i < neu2->size(); i++) 
					{
						sum0 += (*neu1)[i] * (*neu2)[i];
						sum1 += ((*neu1)[i] * (*neu1)[i]);
						sum2 += ((*neu2)[i] * (*neu2)[i]);
					}
  
					sum1 = std::sqrt(sum1);
					sum2 = std::sqrt(sum2);
					ret = std::acos(sum0 / (double)(sum1 * sum2) );

					weight = ret;
         
					break;
				}
				else
				{
					//TODO not implemented
					//e.g. spectral angle similarity
				}

			}  
		} 

		return weight;
	}
  
	bool CompletingGraph::checkMutualfriends(const Node *u, const Node *v, unsigned int&candidate)
  {
		bool found = false;
		candidate = UINT_MAX;
		const std::vector<Node*>& edgeU = u->getEdges();
		const std::vector<Node*>& edgeV = v->getEdges();
		//look for neighbors of V
		for(unsigned int m = 0; m < edgeV.size();m++)
		{
		//compare every neighbor of U with every neighbor of V
			for(unsigned int n = 0; n < edgeU.size();n++)
			{
				if(edgeU.at(n)->getIndex() == edgeV.at(m)->getIndex())
				{
					//we found a common friend of node u and v
					candidate = edgeU.at(n)->getIndex();
					return true;
				}
			}
		}
		return found;
	}


	Mesh::Mesh(const Configuration& c, unsigned int n)
	: CompletingGraph(c, n, c.directed) 
	{
		std::cout << "CONSTRUCT MESH " << std::endl;
		width = c.width;
		height = c.height;
		assert(c.initialDegree == 4 || c.initialDegree == 8);
    noEdges = 2* getSize() - 2 * width;
   
		srand ( time(NULL) );
		std::cout << "Size : " << getSize() <<" nodes, arranged on a " << height << " x " << width  << " lattice " << std::endl;
		std::cout << "Diameter : " << (2* (width-1))<< std::endl;
		std::cout <<"Number of edges: " << noEdges<< std::endl;
		std::cout <<"Neighborhood: " << c.initialDegree<< std::endl;
		std::cout << "MESH CONSTRUCTED " << std::endl;

  }

	void Mesh::nextIter() 
	{
		if (iter == 0) {
			//build periodic/non-periodic 4/8 connected neighborhood
			if(conf.initialDegree == 8)
			{
				unsigned int i, j;
				
				for (i = 0; i < getSize(); ++i) 
				{
					j = i + 1;
					if(conf.periodic)
					{	
						//add edge from right to left
						if(j/width > i/width)
							j-=width;
					}
					else
					{
						if (!(j % width))
							j += (width-1);
					}	
					if (j >= getSize())
					{	
						if(conf.periodic)
						{//add edge from bottom to top
							j = (i+width)%getSize(); 
						}
						else
							continue;
					}	
					
					if (j != i)
					{	
						if(j/width == i/width)//assert nodes are in same row
						{	
							addEdge(i, j);//horizontal edges
							m_dijkstra->addEdge(i, j, 1.0);

						}	
					}	
					j = i + width;
					
					if (j >= getSize())
					{	
						if(conf.periodic)
						{//add edge from bottom to top
							j = (i+width)%getSize(); 
						}
						else
							continue;
					}						

					if (j != i)
					{	
						addEdge(i, j);//vertical edges
						m_dijkstra->addEdge(i, j, 1.0);
					}
					
					//edge bottom right
					j = i + width +1;
					
					if(!(j%(width)==0))
					{	
						if(j > getSize())
						{	
							if(conf.periodic)
							{
								j = ( i + width +1)%getSize();
									
							}
							else
								continue;
						}
						
						if (j != i)
						{	
							addEdge(i,j);//diag down right
							m_dijkstra->addEdge(i, j, sqrt(2.0));
						}	
					}
					else //right border
					{
						if(conf.periodic)
						{
							if(i == (getSize()-1))//last node
								j = 0;
							else
								j = i+1;
							if (j != i)
							{	
								addEdge(i,j);
								m_dijkstra->addEdge(i, j, sqrt(2.0));
							}	
						}	
					}	
					
					//edge botom left
					j = i + width -1;
					
					if((j/width) > (i/width) )
					{	
						if(j >= getSize())
						{	
							if(conf.periodic)
							{
								j = ( i + width -1)%getSize();
							}
							else
								continue;
						}						
						if (j != i)
						{	
							addEdge(i,j);//diag down left
							m_dijkstra->addEdge(i, j, sqrt(2.0));
						}
					}
					else //left border
					{
						if(conf.periodic)
						{
							j = (i+ 2*width -1)%getSize();
							
							if (j != i)
							{	
								addEdge(i,j);
								m_dijkstra->addEdge(i, j, sqrt(2.0));
							}	
						}	
					}	
				}
			}	
			else
			{	
				unsigned int i, j;
				
				for (i = 0; i < getSize(); ++i) 
				{
					j = i + 1;
					
					if(conf.periodic)
					{
						if(j/width > i/width)
							j-=width;
					}
					else
					{
						if (!(j % width))
							j += (width-1);
						
						if (j >= getSize())
							continue;		
					}
					
					if (j != i)
					{
						addEdge(i, j);
						m_dijkstra->addEdge(i, j, 1.0);
					}


					j = i + width;
					if (j >= getSize())
					{	
						if(conf.periodic)
						{
							j = (i+width)%getSize(); 
						}
						else
						{
							continue;
						}	
					}	

					if (j != i)
					{	
						addEdge(i, j);
						m_dijkstra->addEdge(i, j, 1.0);
					}	
				}
			}
		}
		
		CompletingGraph::nextIter();
		if(iter == conf.maxIter)      
		{
			std::cout << "  >> Printing iteration " << iter <<std::endl;
			std::stringstream ss;
			ss << conf.output << "iteration_" <<iter <<".dot";
			std::string fn;
			ss >> fn;
			std::string title = fn;
			//remove .dot
			title.resize((title.size()-4));
			//remove path
			size_t found = title.find_last_of('/');
			title.erase(0,++found);
			std::ofstream file;
			file.open(fn.c_str());
			print(file,title );
			file.close(); 
		}
	}
  
	double Mesh::graphClustering()
	{
		double avgCluster =0.0;
		std::vector<const Node*> nodes = getNodes();
		
		for(unsigned int i = 0; i < nodes.size();i++)
		{ 
			avgCluster += nodes[i]->calcCluster();
		}  
		avgCluster/= (double)nodes.size();
		
		return avgCluster;
	}
  
	void Mesh::initDijkstra()
	{
	 
		m_dijkstra = new fastDijkstra(getSize());
		nextIter();
	}
	
	void Mesh::precomputePaths(bool weighted)
	{
		std::cout << "Precalculating distances " <<std::endl;
		for(unsigned int i = 0; i < (getSize()-1); i++)
		{
			getDistance(i,(i+1),weighted,false);
		}	
		getDistance((getSize()-1),0,weighted,false);
		getDistance((getSize()-1),(getSize()-1),weighted,false);
		std::cout << "Done " <<std::endl;
	}

	void Mesh::scaleDistances(double scale)
	{
		m_dijkstra->scaleDistances(scale);
	}

	unsigned int Mesh::findNextNeighbor(unsigned int index, unsigned int dir) 
	{
		unsigned int res;
		unsigned int row = index / width, col = index % width;
		// to avoid a lifelock, the algorithm can only test the max. possible count of nodes
		unsigned int credit = width;

		unsigned int lastrow = getSize() / width;
		unsigned int lastcol = width-1;

		switch (dir) 
		{
			case 0: col--; break;
			case 1: col++; break;
			case 2: row--; break;
			case 3: row++; break;
		}

		//no neighbors; return UINT_MAX
		if(col > UINT_MAX || col > lastcol || row > UINT_MAX || row > lastrow ||  (row*width + col) >= getSize() )
		{
			return UINT_MAX;
		}
		
		res = row*width + col;
		if((getNode(res) == NULL))
			return UINT_MAX;
      
		if (--credit == 0)
			res = index; 

		return res;
	}
  
	void Mesh::checkNeighbors()
	{
		unsigned int size = getNodes().size();
		unsigned int left,up,right, down;
		for(unsigned int index = 0; index < size;index++)
		{
			left  = findNextNeighbor(index, 0);
			right = findNextNeighbor(index, 1);
			up    = findNextNeighbor(index, 2);
			down  = findNextNeighbor(index, 3);
		}
	}
  
	unsigned int Mesh::addMember(Neuron *n) 
	{
		if (conf.insertion == UNDEFINED) 
		{
			unsigned int index = CompletingGraph::addMember(n);

			if (getNodes().size() == 1)
				return index;

			unsigned int first, second;
			// if we have more than one column in this row,
			// connect in the first dimension
			if ((index < getSize()-1)||(index % width)) 
			{
				first = findNextNeighbor(index, 0);
				second = findNextNeighbor(index, 1);
				// don't delete wrap around
				if ((!conf.directed) && findNextNeighbor(first, 0) != second) 
				{
					Graph::removeEdge(first, second);
				}
				addEdge(first, index);
				addEdge(index, second);
			}
			// if we have more than one row, connect in the second dimension
			if (getSize() > width) 
			{
				first = findNextNeighbor(index, 2);
				second = findNextNeighbor(index, 3);
				// don't delete wrap around
				if ((!conf.directed) && findNextNeighbor(first, 2) != second) 
				{
					Graph::removeEdge(first, second);
				}
				addEdge(first, index);
				addEdge(index, second);
			}

			return index;
    } 
		else 
		{
			std::cerr << "INSERTION MODE NOT IMPLEMENTED YET!" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	unsigned int Mesh::updateNeighborhood(const msi::Node *origin, double depthMax,double learning, const multi_img::Pixel &input)
	{
		double dist = 0.0;
		double sigma = depthMax;
		double weightingFactor =0.0;
		unsigned int res = 0;
		cv::Vec3f color;

		std::vector<const Node*> nodes = getNodes();
		for(unsigned int i = 0; i < nodes.size();i++)
		{	
			Neuron *t = nodes.at(i)->getNeuron();
			dist = getDistance(origin->getIndex(), nodes.at(i)->getIndex(),false);
			if(dist <= depthMax)
			{

				// only for visualization purpose
				if(dist> 4.5){								color[0]=0.6;color[1]=1.0;color[2]=0.6;}
				if(dist <= 4.5 && dist > 4.0){color[0]=1.0;color[1]=0.6;color[2]=0.3;}
				if(dist <= 4.0 && dist > 3.5){color[0]=1.0;color[1]=0.4;color[2]=0.6;}
				if(dist <= 3.5 && dist > 3.0){color[0]=1.0;color[1]=0.0;color[2]=1.0;}
				if(dist <= 3.0 && dist > 2.5){color[0]=0.6;color[1]=0.0;color[2]=1.0;}
				if(dist <= 2.5 && dist > 2.0){color[0]=0.2;color[1]=0.0;color[2]=1.0;}
				if(dist <= 2.0 && dist > 1.5){color[0]=0.0;color[1]=0.2;color[2]=1.0;}
				if(dist <= 1.5 && dist > 1){	color[0]=0.0;color[1]=0.6;color[2]=1.0;}
				if(dist == 1) {								color[0]=0.0;color[1]=1.0;color[2]=1.0;}
				if(dist == 0){								color[0]=0.7;color[1]=0.7;color[2]=0.7;}
				
				weightingFactor = learning * std::exp(- (dist * dist)/(2.0 * sigma * sigma) );
				t->update(input, weightingFactor);
#ifdef WITH_GERBIL_COMMON
				if(!m_msi->empty())
				{  
					t->setSRGB(m_msi->bgr(*t));
				}
#endif
			
			}
		}
		return res;
	} 
  
	double Mesh::getDistance(unsigned int index1, unsigned int index2, bool weighted, bool showPath)
	{
		return m_dijkstra->getDistance(index1,index2,weighted, showPath);
	}
  
  // prints with coordinates to the nodes. use neato -n for good results
	void Mesh::print(std::ostream& o, const std::string& n) const 
	{
		unsigned int i, j;
		
		std::string fillcolor("");

		if (conf.directed)
			o << "digraph ";
		else
			o << "graph ";
		o << n << " {\n"
			"\toverlap=false;\n";
		if(conf.nodes <= 64)
			o << "\tsplines=true;\n";
		//reduce computational effort for neato tool
		else
			o << "\tsplines=false;\n"
			"\tsep = .25;\n";

		std::vector<const Node*> nodes = getNodes();
		for (unsigned int x = 0; x < nodes.size(); x++) 
		{
			i = nodes[x]->getIndex();
			int posx = (i % width) * 150;
			int posy = (i / width) * (-150);
			o << "\t" << i << "[ pos=\"" << posx << "," << posy << "\" ];\n";

			const std::vector<Node*>& edg = nodes[x]->getEdges();

			Neuron *n = nodes[x]->getNeuron(); 
			cv::Vec3f rgb = n->getRGB();
      
			Color c((unsigned int)(rgb[2]* 255.),(unsigned int)(rgb[1]* 255.),(unsigned int)(rgb[0]* 255.));

			fillcolor = c.rgb2hex();
			o << "\t" << i << "[shape= box, style=filled, fillcolor=\"#"<<fillcolor<<"\"]" << ";\n";
			for (j = 0; j < edg.size(); ++j) 
			{
				if (conf.directed) 
				{
					o << "\t" << i << " -> " << edg[j]->getIndex() << ";\n";
				}
				else if (edg[j]->getIndex() >= i) 
				{
					o << "\t" << i << " -- " << edg[j]->getIndex() << ";\n";
				}
			}
		}
		o << "}" << std::endl;
	}
};
