/*
 * @Author: your name
 * @Date: 2021-04-09 16:52:08
 * @LastEditTime: 2021-04-11 10:35:37
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /myNBLDPC/include/GF.h
 */

#ifndef _GF_CUH_
#define _GF_CUH_

#include "define.cuh"
#include "struct.cuh"

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