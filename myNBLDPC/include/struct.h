#ifndef _STRUCT_H_
#define _STRUCT_H_

#include "define.h"
#include <mutex>
#include <thread>
#include <vector>

class CComplex
{
public:
	float Real;
	float Image;
};
class LDPCCode
{
public:
	int maxWeight_checknode;
	int maxWeight_variablenode;
	int GF;
	int Variablenode_num;
	int Checknode_num;
	float rate;
	int bit_length; //比特长度
	int q_bit;		//=log2(GF)
};
class VN
{
public:
	int *linkCNs;
	int *linkCNs_GF;
	int weight;
	float *LLR;
	float *L_ch;
	float **sort_L_v2c;		  //排序过后的
	unsigned **sort_Entr_v2c; //将GF（q）的元素排序，[连接的节点的序号][GF元素]
};
class CN
{
public:
	int *linkVNs;
	int *linkVNs_GF;
	int weight;
	float **L_c2v; //校验节点传给变量节点的值，[连接的节点的序号][各GF对应的LLR]
};
class AWGNChannel
{
public:
	int seed[3]; // 噪声种子
	float sigma;
};

class Simulation
{
public:
	float SNR; // 当前仿真的SNR
	double sumTime;

	long num_Frames;	   // 当前SNR下已经仿真的帧数
	long num_Error_Frames; // 当前SNR下已经错误的帧数
	long num_Error_Bits;   // 当前SNR下错误的总信息比特数
	long Total_Iteration;  // 当前SNR下已经仿真的帧的总迭代次数,用来计算均值
	long num_False_Frames; // 当前SNR下的误判帧数(将错误(信息位有错误比特)的帧判为正确(校验结果正确)的帧)
	long num_Alarm_Frames; // 当前SNR下的虚警帧数(将正确(信息位无错误比特)的帧判为错误(校验结果错误)的帧)

	float FER;		 // =num_Error_Frames/num_Frames
	float BER;		 // =num_Error_Bits/num_Frames
	float AverageIT; // =Total_Iteration/num_Frames
	float FER_False; // =num_False_Frames/num_Frames
	float FER_Alarm; // =num_Alarm_Frames/num_Frames
};

void copyVN(const LDPCCode *H, VN *B, const VN *A);
void freeVN(const LDPCCode *H, VN *A);
void copyCN(const LDPCCode *H, CN *B, const CN *A);
void freeCN(const LDPCCode *H, CN *A);

#endif
