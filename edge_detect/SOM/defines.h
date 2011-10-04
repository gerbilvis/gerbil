#ifndef DEFINES_H
#define DEFINES_H

typedef struct _Position
{
	int x;
	int y;
} Position;

typedef struct _Neuron
{
	double c1;		//first value of codebook vector
	double c2;		//second value of codebook vector
	int l;	//label counter
} Neuron;

typedef struct _Color
{
	char R;
	char G;
	char B;
} Color;

#endif //DEFINES_H