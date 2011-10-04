#include "GraphTest.h"
#include <iostream>
#include <fstream> 
#include <sstream> 
#include <string> 
#include <cstdlib>
#include <time.h>
#include "misc.h"

int main(int argc, char**argv)
{
 
  Configuration conf;
  conf.directed = false;
  conf.completing = true;
  conf.insertion = UNDEFINED;
  conf.nodes = 16;
  conf.startcomp = 0;
  conf.finishcomp = 5;
  

  
  int rnd_delete=5;
  int rnd_num[nodes];
//generate random field  
  /* initialize random seed: */
  srand ( time(NULL) );  
  
  for(int i =0; i < nodes;i++)
  {  
    do{
      rnd_num[i]=rand()%nodes;
      
      for(int i2=0;i2<i;i2++){
        if(rnd_num[i]==rnd_num[i2])rnd_num[i]=-1;
      }
  }while(rnd_num[i]==-1);
  }
  
  std::cout << "# Initialising graph" <<std::endl;
//   vonNeumann graph(5);
//   pso::CompletingGraph graph(conf ,5,false);
//   msi::SW_Circle graph(conf ,nodes);
//   msi::LbestGbest graph(conf ,nodes);
  msi::Mesh graph(conf ,nodes);
  
  const std::string mode("s");
  std::ofstream file;

  for(int i = 0;i< nodes;i++)
  { 
//     Color c((0),(255),0);
    Color c();
    Solution s(i,c);
    graph.update(i, s);
    
  } 
  std::cout << "# Generating graph of size: " << graph.getSize() << std::endl;
  graph.nextIter();

//   std::cout << "# Manipulating graph" <<std::endl;  
//   std::cout << "# Checking neighbors (only for mesh atm)" <<std::endl;  
//   graph.checkNeighbors();
  
  for(int index = 0;index< nodes;index++)
  {  
    std::stringstream ss;
    ss <<"./plot/iteration_" <<index <<".dot";
    std::string fn;
    ss >> fn;
       
//     if(index  < rnd_delete)
    if(index  == 0)      
    {

//       graph.deleteMember(rnd_num[index]);
      graph.deleteMember(4);
      graph.deleteMember(5);
      graph.deleteMember(6);
      graph.deleteMember(7);
      
      Color c;
      const Solution s(5,c);
      graph.insertMember(1,2,s);
      
//       std::cout <<"delete random: " <<rnd_num[index]<<std::endl;
//       graph.deleteMember(rnd_num[index]);
//       graph.addRandomEdge();
      
    }
    file.open(fn.c_str());
    graph.print(file,mode );
    file.close();    
  }  
  
  std::cout << "# Done" <<std::endl;
}
