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



#include "smallWorldGraph.h" 
#include <cassert>

#include <fstream>
#include <iostream>
#include <sstream>

#define LIMIT 10000

using namespace msi;

	SW_Mesh::SW_Mesh(const Configuration& c, unsigned int n)
	: Mesh(c, n), insertround(2), lastinsert(0) 
	{
		std::cout << "CONSTRUCT SMALL WORLD " << std::endl;
		// initialize random seed: 
		srand ( time(NULL) ); 

		assert(c.initialDegree == 4 || c.initialDegree == 8 ); 
    
		initialDegree = c.initialDegree;
		beta = c.beta;
		phi = c.phi;
		sw_model = c.sw_model;
		assert(( sw_model == "BETA") || sw_model == "PHI");
    
    //set topology dependend minimum edgecount
		minimumEdgeCount = (2*n - (2* width));
		if(beta > 0.0 || phi > 0.0)
			m_isSmallWorld = true;
		else
			m_isSmallWorld = false;

	}
  
	bool SW_Mesh::rewire(const Node *n1, const Node *n2)
	{

		bool success = false;
		unsigned int remote_id;

		do
		{
			remote_id = rand() % getSize();
		}
		while( edgeExists(n1->getIndex(),remote_id) && n2->getDegree() < conf.initialDegree);
    
		success = addEdge(n1->getIndex(),remote_id);   
		if(success)
		{
			double weight = m_dijkstra->getEdgeWeight(n1->getIndex(), n2->getIndex());
			
			std::cout << "Rewired n1: " << n1->getIndex() << " ---- n2: " << n2->getIndex() << " with  " <<n1->getIndex()  << " ---- " << remote_id << " weight: " << weight <<std::endl;						
			m_dijkstra->removeEdge(n1->getIndex(), n2->getIndex());
			Graph::removeEdge(n1->getIndex(), n2->getIndex());
			m_dijkstra->addEdge(n1->getIndex(), remote_id,weight);

			return true;
		}
		else
			return false;
	}  
  
	void SW_Mesh::createSmallWorld(double p, sw_type type)
	{
		std::cout << "# Creating Small World Graph with a start degree of " << initialDegree;
		std::cout << " and probability of " << p << std::endl;
    
		std::vector<const Node*> nodes = getNodes();
		std::cout << "#Nodes: " << nodes.size() <<std::endl;
		double avgCluster =0.0;
		avgCluster = graphClustering();
		std::cout << "# Calculated average clustering: " << avgCluster << std::endl;

		int round = 0;
		unsigned int toRewire = 0;
		unsigned int *random_nodes = new unsigned int[getSize()];
    
		std::cout << "# Number of edges: " << Mesh::getEdgeCount() <<std::endl;
    

		double prob = 0.0;
		int actuallyFound = 0;

		std::list<const Node*>::iterator it;
    
		for(unsigned int i = 0; i < getSize();i++)
		{
			const Node *n = getNode(i);
			if(n->getDegree() > conf.initialDegree)
				higherDegree.push_back(n);
			else if(n->getDegree() < conf.initialDegree)
				lowerDegree.push_back(n);
		}	
						
		std::list<const Node*>::iterator iter;

    switch(type)
    {
			case BETA:
				
				const Node *n;
				const Node *e;
				
				for(unsigned int i = 0;i < nodes.size();i++)
				{
					n = getNode(i);
					if(n->getDegree() < conf.initialDegree)
						continue;
					const std::vector<Node*>& edg = n->getEdges();
					for(unsigned int j = 0; j < edg.size(); j ++)
					{	
						
						prob = (double)rand()/RAND_MAX;
						if(prob <= p)
						{  
							e = edg.at(j);
							// avoid edges to be drawn multiple times
							if(e->getIndex() < n->getIndex())
								continue;
							if(rewire(n,e))
								round++;
						}
					}
				}
				std::cout << " # Rewired " << round <<" edges"<< std::endl;

				avgCluster = graphClustering();
        
				std::cout << "# Average clustering of small world: " << avgCluster << std::endl;
				
				break;
    
      case PHI:

        //phi determines percentage of wanted shortcuts
				toRewire = std::floor(phi * (double)getEdgeCount());
				std::cout << "# Rewiring " << toRewire << " edges " <<std::endl;   

				std::cout << "# Begin PHI model " <<std::endl;
				
				for(unsigned int i = 0; i < toRewire;i++)
				{
					//shuffle node IDs
					randomizeNodes(random_nodes);
					unsigned int index =0;
					unsigned int rand_id, friend_id, foreigner_id;
					
          bool found = false;
					// upper limit of search trials to avoid infinite loop
					int trials = 0;
					
					while(!found && (trials <= LIMIT) )
					{
						// pick a random node ID
						rand_id = random_nodes[index];
						// further shuffling 
						index++;
						index = index%getSize();
						const Node *u;
						const Node *v;
						
						// we have nodes possessing larger degree than conf.initialDegree. we prefer those, since u will lose an edge
						if(!higherDegree.empty() )
						{
							// size of higherDegree usually actually does not exceed 1, so ..
							int random;
							if(higherDegree.size() == 1)
								random = 0;
							else// .. this is just for safety
								random = rand() % (higherDegree.size()-1);
							
							it = higherDegree.begin();
							std::advance(it,random);
							// u node taken from list of higherDegree nodes
							
							if(trials > 10)
							{	
								u = getNode(rand_id);
							}
							else
							{
								u = (*it);
							}	
						}	
						else
						{	
							// u node taken by random
							u = getNode(rand_id);
						}	
						if(u->getDegree() < conf.initialDegree)
						{	
							trials++;
							continue;
						}	
						
            const std::vector<Node*>& edgeU = u->getEdges();
            
            //look for neighbors V that share a mutual friend with u and v: u -- commonFriend -- v
						for(unsigned int i = 0; i < edgeU.size();i++)	
						{	
							// also search in random direction 
							int random = rand() % edgeU.size();
							v = edgeU.at(random);
              found = CompletingGraph::checkMutualfriends(u, v, friend_id);
							trials++;
							if(found)
								break;
							
            }
						if(trials == LIMIT)
						{	
							//abort search to avoid infinite loop
							std::cout << "# No mutual friend found after " << trials << " iterations.\nAborted search to avoid infinite loop." <<std::endl;
						}
            
            //if a common friend is found, pick a further node W with randomly distribution, such that U and W do NOT share a common
						// friend : U -- (...) -- W
						if(found)
						{ 
							// we found a node v in the neighborhood of u, s.t. u and v both are direct neighbors of a common friend
// 							std::cout << " Pick node v " << v->getIndex() <<" from immediate neighborhood" <<std::endl;
							round = 0;
							int random;
							const Node *w;
							unsigned int tryLowerDegreeBucket = 0U;
							do
							{
								//shuffle again
								rand_id = random_nodes[index];
								index++;
								index = index%getSize();
								
								// we prefer nodes that have a lower degree than conf.initialDegree, since w will gain an edge
								if(!lowerDegree.empty() && tryLowerDegreeBucket < lowerDegree.size())
								{	
									it = lowerDegree.begin();
									// only one node available 
									if(lowerDegree.size() == 1)
										random = 0;
									else// choose a node by random from the list
										random = rand() % (lowerDegree.size()-1);
									std::advance(it,random);
									w = (*it);
									tryLowerDegreeBucket++;
								}
								else
								{	// choose a node by random from 
									w = getNode(rand_id);
								}	
                round++;
								if(round == LIMIT)
								{
									std::cout << "# No mutual friend found after " << trials << " iterations.\nAborted search to avoid infinite loop part 2." <<std::endl;
								}	
              }
							while(CompletingGraph::checkMutualfriends(v, w, foreigner_id) && (round != LIMIT));
							// we found a node w that does NOT share a common friend with u and w -> shortcut
              foreigner_id = w->getIndex();
							// do the rewiring
							if(addEdge(v->getIndex(), w->getIndex()))
							{	
								double weight = m_dijkstra->getEdgeWeight(u->getIndex(), v->getIndex());
								std::cout << "#	Rewired u: " << u->getIndex() << " ---- " << v->getIndex() << " with v " << v->getIndex() << " ---- " << w->getIndex() << " weight: " << weight <<std::endl;						
								
								m_dijkstra->removeEdge(u->getIndex(), v->getIndex());
								Graph::removeEdge(u->getIndex(), v->getIndex());
								
								m_dijkstra->addEdge(v->getIndex(), w->getIndex(),weight);
								
								if(u->getDegree() == conf.initialDegree && lowerDegree.size() != 0)
								{	
									//u has again conf.initialDegree, so remove u from list with higher degree than wanted
									for(it = higherDegree.begin(); it!= higherDegree.end(); it++)
									{
										if((*it)->getIndex() == u->getIndex())
										{	
											higherDegree.erase(it);
											break;
										}	
									}
								}									
								else if(u->getDegree() < conf.initialDegree)
								{	
									//u has now less than conf.initialDegree, so add u to more preferred nodes
									lowerDegree.push_back(u);
								}	
								
								if( w->getDegree() == conf.initialDegree && lowerDegree.size() != 0) 
								{	//w has again conf.initialDegree, so remove u from list with higher degree than wanted
									for(it = lowerDegree.begin(); it!= lowerDegree.end(); it++)
									{
										if((*it)->getIndex() == w->getIndex())
										{	
											lowerDegree.erase(it);
											break;
										}	
									}									
								}	
								else if( (w)->getDegree() > conf.initialDegree)
								{	// w has now more neighbores than the wanted degree
									higherDegree.push_back(w);
								}									
								actuallyFound++;
							}	
						}  
					}
				}

				std::cout << "# Rewired " << actuallyFound << " of " << toRewire << " edges" <<std::endl;

				avgCluster = graphClustering();
				
				std::cout << "# Average clustering: " << avgCluster <<std::endl;
				
				break;
        
			default:
				std::cout << "# Not implemented yet!" <<std::endl;
				break;
		}
	}
  
	void SW_Mesh::print(std::ostream& o, const std::string& n) const {
		unsigned int i, j;
		std::string fillcolor("000000");

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
			Neuron *n = nodes[x]->getNeuron();
			int posx = (i % width) * 150;
			int posy = (i / width) * (-150);
			o << "\t" << i << "[ pos=\"" << posx << "," << posy << "\" ];\n";

			const std::vector<Node*>& edg = nodes[x]->getEdges();
/**colorize edge count     
//       if(edg.size()==0) fillcolor = "slategray";
//       if(edg.size()==1) fillcolor = "gold";
//       if(edg.size()==2) fillcolor = "orange";
//       if(edg.size()==3) fillcolor = "orangered";
//       if(edg.size()==4) fillcolor = "red";
//       if(edg.size()==5) fillcolor = "crimson";
//       if(edg.size()==6) fillcolor = "green";
//       if(edg.size()==7) fillcolor = "darkgreen";
//       if(edg.size()==8) fillcolor = "lightskyblue";
//       if(edg.size()==9) fillcolor = "blue";
//       if(edg.size()==10) fillcolor = "indigo";
//       if(edg.size()> 10) fillcolor = "grey14";
*/

//colorize spread TODO
			//cv::Vec3f rgb = n->getRGB();
			//Color c((unsigned int)(rgb[2]* 255.),(unsigned int)(rgb[1]* 255.),(unsigned int)(rgb[0]* 255.));
			//fillcolor = c.rgb2hex();
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
  
	bool SW_Mesh::removeRandomEdge()
	{
		unsigned int *random_nodes = new unsigned int[getSize()];
		randomizeNodes(random_nodes);
		const Node *node = getNode(random_nodes[0]);
		if(node->getDegree() >= 2)
		{  
			const std::vector<Node*>& edg = node->getEdges();
			return CompletingGraph::removeEdge(node->getIndex(),edg[0]->getIndex());
		}
		return false;
	}

	void SW_Mesh::nextIter() 
  {
		Mesh::nextIter();
		if (iter == 1) 
		{
			std::cout << " >> SW" << std::endl; 
			if(m_isPeriodic)
				std::cout << " >> Building periodic lattice " << std::endl;
			else std::cout << " >> Building lattice " << std::endl;

			std::cout << " >> " << width << " by " << width <<" Nodes: " << getSize() <<std::endl; 

			//create small world
			if(sw_model == "BETA" && beta > 0.0)
				createSmallWorld(beta, BETA);
			else if(sw_model == "PHI" && phi > 0.0)
				createSmallWorld(phi, PHI);
	}
    
		//print every nth frame
		std::stringstream ss;
		std::string fn;
    
		
		if(iter%(maxIter/10)  == 0 )
			std::cout << "  >> Iteration " << iter <<std::endl;

		if(iter < 3 )      
		{
			std::cout << "  >> Printing iteration " << iter <<std::endl;

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
