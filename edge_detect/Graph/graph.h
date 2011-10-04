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


#ifndef PSO_GRAPH_H
#define PSO_GRAPH_H

// #include "solution.h"
#include "neuron.h"

#include "set"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

namespace msi {

	/** Class describing a node in the neighborhood graph.
	    The node also stores information about the associated Particle. */
	class Node {

		friend class DirNode;
		friend class UnDirNode;

		public:
			/** Default constructor
				@param i index of this node in the graph
			*/
			Node(unsigned int i);
			/** Usual constructor
				@param i index of this node in the graph
				@param n initial Neuron
			*/
// 			Node(unsigned int i, const Solution& sol);
      Node(unsigned int i, Neuron * n);      
			virtual ~Node();

			/*! @return index of this node in the graph */
			inline unsigned int getIndex() const { return index; }
			
			/*! @return best found solution so far (private guide) */
// 			inline const Solution& getBest() const { return best; }
			/*! @return neuron the node carries */
			inline   Neuron* getNeuron() const { return m_neuron; }

			/** Gives distance between two private guides.
				@param to Node that contains the other private guide
				@return Euclidean distance in search space
			 */
// 			double getDistance(const std::vector<double>& to) const;

			/*! @return count of adjacent nodes / direct predecessors and successors */
			virtual inline unsigned int getDegree() const { return edges.size(); }

			/*! @return vector of adjacent nodes / direct successors */
			inline const std::vector<Node*>& getEdges() const {
				return edges;
			}

			/**
			 * Update current neuron
			 * @param new_neuron New neuron to replace the current
			 * @return true on success //TODO maybe void
			 */
			bool update( Neuron *new_neuron);

			/**
			 * Add a new edge
			 * @return true if there wasn't already an edge
			 */
			virtual bool addEdge(Node* to)=0;
			/**
			 * Remove an existing edge
			 * @return false if edge wasn't there anyways
			 */
			virtual bool removeEdge(Node* to)=0;

			/** Delete all edges to and from this node
			 *  @return count of edges removed
			 */
			virtual unsigned int disconnect()=0;

			/**
			 * Test for an edge
			 * @param from head node
			 * @param to tail node
			 * @return true if there is a (from, to)
 			 */
			static bool edgeExists(const Node* from, const Node* to);

			/*! @return cluster index the node is currently associated with */
			inline unsigned int getCluster() const { return cluster; }
			/*! @param c cluster index the node is currently associated with */
			inline void setCluster(unsigned int c) { cluster = c; }

			/**
			 * Helper function to edit adjacence lists
			 * @param vec Vector holding the Node pointers
			 * @param p Node to seek and remove
			 * @return false if p wasn't found
 			 */
			inline static bool remove(std::vector<Node*>& vec, Node *p) {
				std::vector<Node*>::iterator it;
				for (it = vec.begin(); it != vec.end(); ++it)
				if (*it == p) {
					vec.erase(it);
					return true;
				}
				return false;
			}
			
			double calcCluster()const;

		protected:
			std::vector<Node*> edges;
		private:
			unsigned int index;
			unsigned int cluster;
      
			Neuron *m_neuron;

      
	};

	class DirNode;
	/** Node with edges, for usage inside an undirected graph.
		It would also work to mix directed Nodes with undirected ones. */
	class UnDirNode : public Node {
		public:
			/** Default constructor
				@param i index of this node in the graph
			*/
			UnDirNode(unsigned int i);
			/** Usual constructor
				@param i index of this node in the graph
				@param sol initial Neuron
			*/
			UnDirNode(unsigned int i, Neuron *n);
			~UnDirNode();

			/** Transform directed Node to undirected.
			    @param src DirNode to be transformed to UndirNode;
				           src will be deleted!
			*/
			UnDirNode(DirNode* src);
			bool addEdge(Node* to);
			bool removeEdge(Node* to);

			unsigned int disconnect();
	};

	/** Node with arcs, for usage inside a directed graph */
	class DirNode : public Node {
		friend class UnDirNode;
		public:
			/** Default constructor
				@param i index of this node in the graph
			*/
			DirNode(unsigned int i);
			/** Usual constructor
				@param i index of this node in the graph
				@param sol initial Neuron
			*/
			DirNode(unsigned int i, Neuron *n);
			~DirNode();

			// we want incoming+outgoing degree here
			inline unsigned int getDegree() const
			{ return edges.size() + incoming.size(); }

			/** Transform undirected Node to directed.
			    @param src DirNode to be transformed to UndirNode;
				           src will be deleted!
			*/
			DirNode(UnDirNode* src);
			bool addEdge(Node* to);
			bool removeEdge(Node* to);

			unsigned int disconnect();

		private:
			/** Vector holding all predecessors of this Node */
			std::vector<Node*> incoming;
	};

	/** Manages the basic neighborhood topology.
	 * Operations on the topology should always be done using this object.
	 * Derived classes don't get direct access to the Nodes, instead, they can use
	 * a Getter to read them and traverse the graph, and specific methods to operate on them.
	 * For example, if edges are added or removed, the Graph can track the total edge count.
	 */
	class Graph {

		public:
			/** Default constructor.
				No Node objects are created, only space is reserved for them. They are created
				on the first call of update().
			    @param n Initial count of Nodes
				@param dir Wether this is a directed graph or not. If it's directed, all Nodes
					created in the future will be DirNode.
			*/
			Graph(unsigned int n, bool dir);

			/**
			 * Give the size of the reserved space for Nodes, not the actual Swarm size.
			 * @note To retrieve the count of nodes present in the graph, use getNodes().size()
			 * @return max. index value ever used + 1
			 **/
			inline unsigned int getSize() const { return nodes.size(); }

			/**
			 * Give a Node by its index.
			 * @param idx index of the Node wanted
			 * @return corresponding Node
			 */
			inline const Node* getNode(unsigned int idx) const
			{	
        assert(idx < nodes.size());
				return nodes[idx]; }
			/**
			 * Give all current existing nodes in the graph.
			 * @return vector of valid Node pointers
			 **/
			std::vector<const Node*> getNodes() const;

			/**
			 * Print the graph in "dot" format.
			 * @param o stream to write to
			 * @param n name of graph
			 */
			virtual void print(std::ostream& o, const std::string& n) const;

			/**
			 * Reserve space for new Nodes (to be initialised by update()).
			   @param count count of new Nodes
			   @return new size of Node datafield
			 */
			virtual unsigned int add(unsigned int count);
			/**
			 * Remove a Node by its index.
			 * The Node will be disconnected from the Graph and free'd.
			 * @param index index of the Node to be removed
			 */
			virtual void remove(unsigned int index);

			/**
			 * Update current neuron
			 * @param index index of the Node in question
			 * @param sol New to replace the current one
			 * @return true on success
			 */
			bool update(unsigned int index,Neuron *n);
      
			/*! Set new cluster index for a Node
				@param index index of the Node in question
				@param c cluster index the node is currently associated with */
			inline void setCluster(unsigned int index, unsigned int c)
			{ 	assert(index < nodes.size());
				if (nodes[index] != NULL)	nodes[index]->setCluster(c); }

			/*! @return Total count of edges/arcs in the graph */
			inline unsigned int getEdgeCount() const { return edgeCount; }
			bool edgeExists(unsigned int from, unsigned int to) const;
			/**
			 * Add a new edge/arc
			 * @param from head of the edge/arc
			 * @param to tail of the edge/arc
			 * @return true if there wasn't already an edge
			 */
			bool addEdge(unsigned int from, unsigned int to);
			/**
			 * Remove an existing edge
			 * @param from head of the edge/arc
			 * @param to tail of the edge/arc
			 */
			void removeEdge(unsigned int from, unsigned int to);

			/**
			 * Removes all edges of this graph.
			 * Disconnects all Nodes.
			 **/
			void removeAllEdges();
      
			//! Increase the graph size
      unsigned int addMember(Neuron *n);

			//! Decrease the graph size			
      Neuron* deleteMember(unsigned int index);
      
			//! Next iteration 
      void nextIter(){iter++;}

    protected:
      unsigned int iter;
      
		private:
			std::vector<Node*> nodes;
			bool directed;
      int edgeCount;
			
	};
};

#endif
