#include "dijkstra.h" 

using namespace dijkstra;


Dijkstra::Dijkstra()
{

}
void Dijkstra::getShortestPath(Node *destination)
{
  nodes = m_nodes;
  
  while (nodes.size() > 0)
  {
    Node* smallest = ExtractSmallest(nodes);
    std::vector<Node*>* adjacentNodes =  AdjacentRemainingNodes(smallest);
    const unsigned int size = adjacentNodes->size();
    for (unsigned int i=0; i<size; ++i)
    {
      Node* adjacent = adjacentNodes->at(i);
      float distance = Distance(smallest, adjacent) +
        smallest->distanceFromStart;
      
      if (distance < adjacent->distanceFromStart)
      {
        
        adjacent->distanceFromStart = distance;
        adjacent->previous = smallest;
        destination = adjacent;
        
//         std::cout << "Destination distance : " <<destination->distanceFromStart <<std::endl;
      }
    }
    delete adjacentNodes;
  }
  
//   Node* previous = destination;
//   if(!previous)return UINT_MAX;  
 
//   return destination->distanceFromStart;
}

void Dijkstra::allocate(unsigned int size)
{
  m_nodes.reserve(size);
}

void Dijkstra::addNode(Node *n)
{
  m_nodes.push_back(n);
  
}

void Dijkstra::addEdge(Edge *e)
{
  if(e->node1->id > e->node2->id)
  {
    //swap
    Node*t = e->node1;
    e->node1 = e->node2;
    e->node2 = t;
  
  }  
  edges.push_back(e);
}

void Dijkstra::cleanUpEdges()
{
//TODO remove multiple edges
}

//actually: set starting node distance to zero
void Dijkstra::setDistance(Node *n, float dist)
{
  for(unsigned int i = 0; i < m_nodes.size() ;i++)
  {  
    if(m_nodes.at(i)->id == n->id)
    {
      m_nodes.at(i)->distanceFromStart = 0.0;
    }  
  }  
}



// Find the node with the smallest distance,
// remove it, and return it.
Node* Dijkstra::ExtractSmallest(std::vector<Node*>& nodes)
{
  unsigned int size = nodes.size();
  if (size == 0) return NULL;
  unsigned int smallestPosition = 0;
  Node* smallest = nodes.at(0);
  for (unsigned int i=1; i<size; ++i)
  {
    Node* current = nodes.at(i);
    if (current->distanceFromStart <
      smallest->distanceFromStart)
    {
      smallest = current;
      smallestPosition = i;
//       std::cout << "smallest position " <<smallestPosition <<std::endl;
    }
  }
  nodes.erase(nodes.begin() + smallestPosition);
  return smallest;
}

// Return all nodes adjacent to 'node' which are still
// in the 'nodes' collection.
std::vector<Node*>* Dijkstra::AdjacentRemainingNodes(Node* node)
{
  std::vector<Node*>* adjacentNodes = new std::vector<Node*>();
  const unsigned int size = edges.size();
//   std::cout << "edges size " << edges.size() <<std::endl;
  for(unsigned int i=0; i<size; ++i)
  {
    Edge* edge = edges.at(i);
//     std::cout << "Checking edge "<< edge->node1->id << " ---- " <<edge->node2->id  << " Node id: "<< node->id << std::endl;
    Node* adjacent = NULL;
    if (edge->node1->id == node->id)
    {
      adjacent = edge->node2;
    }
    else if (edge->node2->id == node->id)
    {
      adjacent = edge->node1;
    }
    if (adjacent && Contains(m_nodes, adjacent))
    {
      adjacentNodes->push_back(adjacent);
    }
  }
  return adjacentNodes;
}
// Return distance between two connected nodes
float Dijkstra::Distance(Node* node1, Node* node2)
{
  const unsigned int size = edges.size();
  for(unsigned int i=0; i<size; ++i)
  {
    Edge* edge = edges.at(i);
    if (edge->Connects(node1, node2))
    {
      return edge->distance;
    }
  }
  return -1; // should never happen
}
// Does the 'nodes' vector contain 'node'
bool Dijkstra::Contains(std::vector<Node*>& nodes_vec, Node* node)
{
  const unsigned int size = nodes_vec.size();
  for(unsigned int i=0; i<size; ++i)
  {
    if (node == nodes_vec.at(i))
    {
      return true;
    }
  }
  return false;
}

void Dijkstra::PrintShortestRouteTo(Node* destination)
{
  Node* previous = destination;
  std::cout << "Distance from start: " << destination->distanceFromStart << std::endl;
  while (previous)
  {
    std::cout << previous->toString() << " ";
    previous = previous->previous;
  }
  std::cout << std::endl;
}

void Dijkstra::resetDistances()
{
  for(unsigned int i = 0; i < m_nodes.size();i ++)m_nodes[i]->distanceFromStart = FLT_MAX;
}

float Dijkstra::getDiameter()
{
  float diameter = FLT_MAX;
  float largest = 0.0f;
  for(unsigned int i = 0; i < m_nodes.size(); i++)
  {
    for(unsigned int j = 0; j < m_nodes.size(); j++)
    {
      diameter = getDistance(i, j);
      if(diameter > largest)largest = diameter;
    }  
  }

  return largest;
}


float Dijkstra::getDistance(unsigned int indexA, unsigned int indexB)
{
//   std::cout <<"fu"<<std::endl;
  assert(indexA < m_nodes.size() && indexB < m_nodes.size());

  Node *a = getNode(indexA);
  Node *b = getNode(indexB);
  resetDistances();
  setDistance(a, 0.0);
//   if(a->id == b->id)return 0U;
  getShortestPath(b);
  
//   PrintShortestRouteTo(b);
//   std::cout << "Distance " << a->id << " ---- " <<b->id << " : " << b->distanceFromStart << std::endl;
  return b->distanceFromStart;
}


Dijkstra::~Dijkstra(){}
