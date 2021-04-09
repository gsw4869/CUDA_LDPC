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

typedef unsigned u32;
#define Matrixfile "BDS.576.288.GF.64.txt"
#define Constellationfile "./Constellation/GRAY_64QAM.txt"

/*LDPC码相关参数*/


// #define	J		         			4									// 行分块个数
// #define	L       					24									// 列分块个数
// #define	Z       					96								    // 分块矩阵维数,其中每个分块矩阵的维数为Z_define*Z_define
// #define	CW_Len      				2304								// 码字长度,用于GPU计算中
// #define	msgLen      				1920								// 信息位长度,用于GPU计算中
// #define	parLen          			384         						// 校验位长度,用于GPU计算中
// #define	PN_Message          		0									// 0--全零序列; 1--PN序列(注意,本程序只支持全零序列)
// #define minWeight_checknode			4
// #define maxWeight_checknode			25
// #define minWeight_variablenode		2
// #define	maxWeight_variablenode		15
// #define rate                        ((float)msgLen/CW_Len)


/*LDPC译码器相关参数*/
#define	maxIT				50									// LDPC译码器最大迭代次数.其中对Q值赋初值用了一次迭代
//#define	opt_R				(0.83)								// NMS算法中的修正因子.浮点译码器
#define	decoder_method		0									// 译码算法:0->NMS;1->BP

/*AWGN����*/
#define	ix_define					173
#define	iy_define					173
#define	iz_define					173

#define	Add_noise			1									// 0--No; 1--Yes
#define	snrtype				0									// 0--Eb/No; 1--Es/No

/*�������*/
#define	startSNR		    5
#define	stepSNR				1
#define	stopSNR				13.0

#define	leastErrorFrames		50									// 最少错误帧数
#define	leastTestFrames		    10000								// 最少仿真帧数
#define	displayStep      		40960								// 定义将译码结果写入相应txt文件的频率

/*CUDA c��Ӧ����*/
#define MaxThreadPerBlock   	1024								// 针对GeForce GTX 1050而言.
#define PI (3.1415926)
#define CPU_GPU		1									// 采用CPU还是GPU进行译码:0->CPU;1->GPU
#define Num_Frames_OneTime    4096							// 一次同时处理的帧数
#define Message_CW	0									// 提前终止和统计时只看信息位还是看整个码字:0->只看信息位;1->看整个码字

#endif