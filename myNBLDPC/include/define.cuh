#ifndef _DEFINE_H_
#define _DEFINE_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//#include <conio.h>
#include <string.h>
#include <memory.h>
#include <time.h>
//#include <direct.h>
#include "struct.cuh"
#include "Simulation.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define Matrixfile "BDS.576.288.GF.64.txt"
#define Constellationfile "./Constellation/BPSK.txt"
#define n_QAM 2
#define GFQ 64
#define maxdc 4
#define maxdv 2

/*LDPC译码器相关参数*/
#define maxIT 50 // LDPC译码器最大迭代次数.其中对Q值赋初值用了一次迭代
//#define	opt_R				(0.83)								// NMS算法中的修正因子.浮点译码器
#define decoder_method 0 // 译码算法:0->NMS;1->BP

/*AWGN参数*/
#define ix_define 173
#define iy_define 173
#define iz_define 173

#define Add_noise 1 // 0--No; 1--Yes
#define snrtype 0   // 0--Eb/No; 1--Es/No

/*仿真参数*/
#define startSNR 0
#define stepSNR 0.5
#define stopSNR 10

#define leastErrorFrames 50 // 最少错误帧数
#define leastTestFrames 10  // 最少仿真帧数
#define displayStep 1       // 定义将译码结果写入相应txt文件的频率

/*CUDA c相应参数*/
#define MaxThreadPerBlock 1024 // 针对GeForce GTX 1050而言.
#define PI (3.1415926)
#define CPU_GPU 1               // 采用CPU还是GPU进行译码:0->CPU;1->GPU
#define Num_Frames_OneTime 4096 // 一次同时处理的帧数
#define Message_CW 0            // 提前终止和统计时只看信息位还是看整个码字:0->只看信息位;1->看整个码字

#endif