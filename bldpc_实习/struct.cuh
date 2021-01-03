#ifndef _STRUCT_H_
#define _STRUCT_H_

#include "define.cuh"

typedef struct
{
	int iteraTime;		// ��ǰ֡����������õĵ�������,ͳ����
}LDPCCode;
typedef struct
{
	int seed[3];		// ��������
	float sigma;
}AWGNChannel;

typedef struct
{
	float SNR;			// ��ǰ�����SNR

	long num_Frames;		// ��ǰSNR���Ѿ������֡��
	long num_Error_Frames;	// ��ǰSNR���Ѿ������֡��
	long num_Error_Bits;	// ��ǰSNR�´��������Ϣ������
	long Total_Iteration;	// ��ǰSNR���Ѿ������֡���ܵ�������,���������ֵ
	long num_False_Frames;	// ��ǰSNR�µ�����֡��(������(��Ϣλ�д������)��֡��Ϊ��ȷ(У������ȷ)��֡)
	long num_Alarm_Frames;	// ��ǰSNR�µ��龯֡��(����ȷ(��Ϣλ�޴������)��֡��Ϊ����(У��������)��֡)
	
	float FER;				// =num_Error_Frames/num_Frames
	float BER;				// =num_Error_Bits/num_Frames
	float AverageIT;		// =Total_Iteration/num_Frames
	float FER_False;		// =num_False_Frames/num_Frames
	float FER_Alarm;		// =num_Alarm_Frames/num_Frames
	
}Simulation;

#endif