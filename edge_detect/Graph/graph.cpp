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


#include "graph.h"
#include <cmath>

using namespace msi;

	/* ********* NODE ********** */

	Node::Node(unsigned int i) : index(i) 
	{}

	Node::Node(unsigned int i, Neuron* n)
	: index(i), m_neuron(n) 
	{}

	Node::~Node() {}

	bool Node::update(Neuron *n) 
	{
		m_neuron = n;
		return true;
	}

	bool Node::edgeExists(const Node* from, const Node* to) 
	{
		assert(from != NULL && to != NULL);
		return (std::find(from->edges.begin(), from->edges.end(), to) != from->edges.end());
	}
  
	double Node::calcCluster()const
	{
  //alternative formulation for clustering coefficient to Watts (without faculty)
	// Cluster coefficient :C_i = (2* n)/(k_i(k_i-1))
	// n: number of actual edges in neighborhood of node i
	// k_i: number of neighbors of node i
    
		unsigned int possibleEdges = 0U;
		unsigned int n = 0U;
		
		if(edges.size() == 1)
			return 1.0;

		for(unsigned int i = 0; i < edges.size();i++)
		{
			for(unsigned int j = i; j < edges.size(); j++)
			{
				if(edgeExists(edges[i],edges[j]))
				{
					n++;
				}	
			}  
		}
		possibleEdges = (getDegree() * (getDegree()-1));
		
		return ( (2.0* (double)n)/((double)possibleEdges));
	}

	/* ******* UNDIRNODE ******** */

	UnDirNode::UnDirNode(unsigned int i) : Node(i) {}

	UnDirNode::UnDirNode(unsigned int i, Neuron *n) : Node(i, n) {}

	UnDirNode::~UnDirNode() {
		disconnect();
	}

	UnDirNode::UnDirNode(DirNode* src) : Node(src->index) 
	{
		m_neuron = src->m_neuron;
		cluster = src->cluster;

		// becoming undirected means every incoming connection leads to an outgoing one, and vice versa
		for (unsigned int i = 0; i < src->edges.size(); ++i) 
		{
			addEdge(src->edges[i]);
		}
		for (unsigned int i = 0; i < src->incoming.size(); ++i) 
		{
			addEdge(src->incoming[i]);
		}
		// also, the old object has to be killed (in OUR design ;)
		delete src;
	}

	bool UnDirNode::addEdge(Node* to) 
	{
		if (std::find(edges.begin(), edges.end(), to) == edges.end()) 
		{
			edges.push_back(to);
			to->edges.push_back(this);
			return true;
		}
		return false;
	}

	bool UnDirNode::removeEdge(Node* to) 
	{
		remove(edges, to);
		return remove(to->edges, this);
	}

	unsigned int UnDirNode::disconnect() 
	{
		unsigned int count = edges.size();

		for (unsigned int i = 0; i < edges.size(); ++i) 
		{
			remove(edges[i]->edges, this);
		}
		edges.clear();
		return count;
	}

	/* *******  DIRNODE  ******** */

	DirNode::DirNode(unsigned int i) : Node(i) {}

	DirNode::DirNode(unsigned int i, Neuron *n) : Node(i, n) {}

	DirNode::~DirNode() {
		disconnect();
	}

	DirNode::DirNode(UnDirNode* src) : Node(src->index) {
		m_neuron = src->m_neuron;
		cluster = src->cluster;

		// becoming directed means we have to see previous edges as a combination of two arcs
		for (unsigned int i = 0; i < src->edges.size(); ++i) 
		{
			addEdge(src->edges[i]);
			src->edges[i]->addEdge(this);
		}
		// also, the old object has to be killed (in OUR design ;)
		delete src;
	}

	bool DirNode::addEdge(Node* to) 
	{
		if (std::find(edges.begin(), edges.end(), to) == edges.end()) 
		{
			edges.push_back(to);
			DirNode* t = dynamic_cast<DirNode*>(to);
			if (t != NULL)
				t->incoming.push_back(this);
		
			return true;
		}
		return false;
	}

	bool DirNode::removeEdge(Node* to) 
	{
		DirNode* t = dynamic_cast<DirNode*>(to);
		if (t != NULL)
			remove(t->incoming, this);
	
		return remove(edges, to);
	}

	unsigned int DirNode::disconnect() {
		unsigned int count = edges.size() + incoming.size();
		for (unsigned int i = 0; i < edges.size(); ++i) {
			DirNode* t = dynamic_cast<DirNode*>(edges[i]);
			if (t != NULL)
				remove(t->incoming, this);
		}
		edges.clear();
		for (unsigned int i = 0; i < incoming.size(); ++i) 
		{
			remove(incoming[i]->edges, this);
		}
		incoming.clear();
		return count;
	}

	/* ********* GRAPH ********** */

	Graph::Graph(unsigned int n, bool dir) : nodes(n), directed(dir), edgeCount(0) 
	{
		Node* X = NULL;
		iter = 0;
		std::fill(nodes.begin(), nodes.end(), X);
	}

	void Graph::print(std::ostream& o, const std::string& n) const 
	{
		unsigned int i, j;

		if (directed) 
			o << "digraph ";
		else 
			o << "graph ";

		o << n << " {\n"
			 "\toverlap=false;\n"
			 "\tsplines=true;\n"
			 "\tsep = .25;\n";

		for (i = 0; i < nodes.size(); ++i) 
		{
			if (nodes[i] == NULL)
				continue;

			const std::vector<Node*>& edg = nodes[i]->getEdges();

			for (j = 0; j < edg.size(); ++j) 
			{
				if (directed) 
					o << "\t" << i << " -> " << edg[j]->getIndex() << ";\n";
				else if (edg[j]->getIndex() >= i)
					o << "\t" << i << " -- " << edg[j]->getIndex() << ";\n";
			}
		}

		o << "}" << std::endl;
	}

	std::vector<const Node*> Graph::getNodes() const {
		std::vector<const Node*> avail;
		for (unsigned int i = 0; i < nodes.size(); ++i)
			if (nodes[i] != NULL)
				avail.push_back(nodes[i]);
		return avail;
	}


	bool Graph::edgeExists(unsigned int from, unsigned int to) const 
	{
		assert(from < nodes.size() && to < nodes.size());
		assert(nodes[from] != NULL && nodes[to] != NULL);
	
		return Node::edgeExists(nodes[from], nodes[to]);
	}

	bool Graph::addEdge(unsigned int from, unsigned int to) 
	{
		assert(from < nodes.size() && to < nodes.size());
		assert(nodes[from] != NULL && nodes[to] != NULL);
		if (from == to)
			return false;
		if (!nodes[from]->addEdge(nodes[to]))
			return false;

		edgeCount++;
		return true;
	};

	void Graph::removeEdge(unsigned int from, unsigned int to) 
	{
		assert(from < nodes.size() && to < nodes.size());
		assert(nodes[from] != NULL && nodes[to] != NULL);
		edgeCount--;
		nodes[from]->removeEdge(nodes[to]);
	};

	unsigned int Graph::add(unsigned int count) 
	{
		Node* X = NULL;
		nodes.resize(nodes.size() + count, X);
		return nodes.size()-1;
	}
	
	unsigned int Graph::addMember(Neuron *n)
	{
		unsigned int index = add(1);
		update(index, n);

		return index;
  }
    
	Neuron* Graph::deleteMember(unsigned int index)
	{
		assert(index < nodes.size() && nodes[index] != NULL);
		Neuron *n = nodes[index]->getNeuron();

		remove(index);
		nodes.resize(nodes.size());
    
		std::cout << "Deleted node " << index <<"\n" << std::endl;
   
		return n;
	}  

	void Graph::remove(unsigned int i) 
	{
		assert(i < nodes.size());
    
		if (nodes[i] != NULL) 
		{
			edgeCount -= nodes[i]->disconnect();

			delete nodes[i];
			nodes[i] = NULL;
		}
	}

	bool Graph::update(unsigned int i, Neuron* n) 
	{
		assert(i < nodes.size());

		if (nodes[i] == NULL) 
		{
			if (directed)
				nodes[i] = new DirNode(i, n);
			else
				nodes[i] = new UnDirNode(i, n);
			return true;
		}
		else 
			return nodes[i]->update(n);
	}

	void Graph::removeAllEdges() 
	{
		for (unsigned int i = 0; i < nodes.size(); i++) 
		{
			if (nodes[i] != NULL) 
				nodes[i]->disconnect();
		}
		edgeCount = 0;
	}
