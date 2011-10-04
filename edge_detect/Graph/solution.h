#ifndef SOLUTION_H
#define SOLUTION_H

#include <string>
#include <vector>
#include "neuron.h"
#include "misc.h"

class Solution
{
  public: 
    Solution(int object);
    Solution(int object, Color color);    
    Solution(int const *p, int o);
    Solution(int * const *p, const int &o);
    Solution();
    Color getColor();
    void setColor(Color c);
    virtual ~Solution();
    
//   private:  
    int obj;
    int *pos;
    Color c;
};

#endif
