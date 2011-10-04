CC      = g++
CCFLAGS = -Wall -c -g
LD      = g++

RM           = rm -f
MAKE         = make
OBJS         = DijkstraTest.o dijkstra.o 
#data_traits.o pixel.o multi_img.o neuron.o 
TARGET       = DijkstraTest

#CV_INCLUDE 	= /usr/include/opencv
#CV_LIB 		= /usr/lib

#MS_EDGES_INCLUDE    	= ../
#MS_EDGES_LIB    	= ../../../core/core/ms_edges/

#VOLE_COMMON_INCLUDE 	= ../../../core/common/
#VOLE_COMMON_LIB 	= ../../../core/core/common/
#VOLE_GERBIL_LIB 	= ../../../core/core/gerbil/common/

#DIR_LIBS = -L$(CV_LIB) \
#	   -L$(VOLE_COMMON_LIB) \
#	   -L$(VOLE_GERBIL_LIB) \
 #	   -L$(MS_EDGES_LIB) 
	    

DIR_INCL = -I./ 
#           -I$(CV_INCLUDE)\
#           -I$(VOLE_COMMON_INCLUDE) \
#           -I$(MS_EDGES_INCLUDE)

#LIBS = -lcv \
#       -lhighgui \
#       -lms_edge_detection \
#       -lvole_common \
#       -lgerbil_common 

#SYSLIBS = -lpthread \
#          -lrt \
#	  -lasound\
#	  -lz

LDFLAGS = $(DIR_LIBS) $(LIBS) $(SYSLIBS) 

$(TARGET): $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

%.o : ./%.cpp
	$(CC) $(CCFLAGS) $(DIR_INCL) $(LDFLAGS) $<

clean:
	$(RM) *.o GraphTest
