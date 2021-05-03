#ifndef _GF_CUH_
#define _GF_CUH_

#include "define.h"
#include "struct.h"

extern unsigned **TableAdd;
extern unsigned **TableMultiply;
extern unsigned *TableInverse;

unsigned **malloc_2(int xDim, int yDim);
float **malloc_2_float(int xDim, int yDim);
int GFAdd(int ele1, int ele2);
int GFMultiply(int ele1, int ele2);
int GFInverse(int ele);

bool GFInitial(int GFq);

#endif
