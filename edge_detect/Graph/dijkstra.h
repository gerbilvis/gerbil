#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include <vector>
#include <iostream>
#include <limits.h>
#include <string>
#include <sstream>

#include "cv.h"

/**
*Code used and modified from
*http://www.reviewmylife.co.uk/blog/2008/07/15/
*/

namespace dijkstra
{
  
  class Node;
  class Edge;
  

  
  class Node
  {
  
  public:
    Node(unsigned int id)
      : id(id), previous(NULL),
      distanceFromStart(FLT_MAX)
    {
//       nodes.push_back(this);
    }
  public:
    inline std::string toString()
    {
      std::stringstream ss;
      ss << id;
      return ss.str();
    }
    
    unsigned int id;
    Node* previous;
    float distanceFromStart;
  };
  
  class Edge
  {
    
  public:
    Edge(Node* node1, Node* node2, float distance)
      : node1(node1), node2(node2), distance(distance)
    {
//       edges.push_back(this);
    }
    
    inline std::string toString()
    {
      std::stringstream ss;
      ss << node1->id << " " << node2->id << " " <<distance;
      return ss.str();
    }
    
//     inline void allToString()
//     {
//       for(unsigned int i =0; i < edges.size();i++)
//         std::cout << edges.at(i)->toString() <<std::endl;
//     }
    
    bool Connects(Node* node1, Node* node2)
    {
      return (
        (node1->id == this->node1->id &&
        node2->id == this->node2->id) ||
        (node1->id == this->node2->id &&
        node2->id == this->node1->id));
    }
  public:
    Node* node1;
    Node* node2;
    float distance;
  };
  
    class Dijkstra
  {

  public:
      
    Dijkstra();
    virtual ~Dijkstra();
    
    void addNode(Node* n);
    void addEdge(Edge* e);
    void PrintShortestRouteTo(Node* destination);
    void getShortestPath(Node* destination);
    Node * first(){return m_nodes[0];}
    Node * last(){return m_nodes[(m_nodes.size()-1)];}
    void setDistance(Node *n, float dist);
    float getDiameter();
    float getDistance(unsigned int indexA, unsigned int indexB);
    void resetDistances();
    void allocate(unsigned int size);
    void setAdjacenzMatrix(cv::Mat1f weights);
    unsigned int size(){return m_nodes.size();}
    
  private:
      
      Node *ExtractSmallest(std::vector<Node*>& nodes);
      std::vector<Node*>* AdjacentRemainingNodes(Node *n);
      float Distance(Node *smalles, Node*adjacent);
      bool Contains(std::vector<Node*>& nodes, Node* node);
      
      
      void cleanUpEdges();
      inline Node* getNode(unsigned int id)
      {
        Node *ret = NULL;
        for(unsigned int i = 0; i < m_nodes.size();i++)
        {
          if(m_nodes.at(i)->id == id)
            ret =  m_nodes.at(i);
        }
        return ret;
      }
      
      std::vector<Node*> m_nodes;
      std::vector<Node*> nodes;
      std::vector<Edge*> edges;      
      
  };

}//end dijkstra  

#endif