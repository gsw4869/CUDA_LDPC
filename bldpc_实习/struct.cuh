#ifndef _STRUCT_H_
#define _STRUCT_H_

#include "define.cuh"

typedef struct
{
	int iteraTime;		// 当前帧完成译码所用的迭代次数,统计用
}LDPCCode;
typedef struct
{
	int seed[3];		// 噪声种子
	float sigma;
}AWGNChannel;

typedef struct
{
	float SNR;			// 当前仿真的SNR

	long num_Frames;		// 当前SNR下已经仿真的帧数
	long num_Error_Frames;	// 当前SNR下已经错误的帧数
	long num_Error_Bits;	// 当前SNR下错误的总信息比特数
	long Total_Iteration;	// 当前SNR下已经仿真的帧的总迭代次数,用来计算均值
	long num_False_Frames;	// 当前SNR下的误判帧数(将错误(信息位有错误比特)的帧判为正确(校验结果正确)的帧)
	long num_Alarm_Frames;	// 当前SNR下的虚警帧数(将正确(信息位无错误比特)的帧判为错误(校验结果错误)的帧)
	
	float FER;				// =num_Error_Frames/num_Frames
	float BER;				// =num_Error_Bits/num_Frames
	float AverageIT;		// =Total_Iteration/num_Frames
	float FER_False;		// =num_False_Frames/num_Frames
	float FER_Alarm;		// =num_Alarm_Frames/num_Frames
	
}Simulation;

#endif