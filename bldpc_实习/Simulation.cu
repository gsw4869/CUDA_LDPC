#include "Simulation.cuh"
#include "LDPC_Encoder.cuh"
#include "LDPC_Decoder.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
* ���溯��
* AWGN:AWGNChannel������������������ӵ�
* 
*/
void Simulation_GPU(AWGNChannel* AWGN, float* sigma_GPU, Simulation* SIM, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode)
{
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	int Num_Device;
	int* CodeWord;
	int* CodeWord_GPU;			// ����������������,Num_Frames_OneTime_define֡,������GPU�ڴ���
	float* Channel_Out_GPU;		// Num_Frames_OneTime_define֡���ݾ���AWGN�ŵ���Ľ��,������GPU�ڴ���
	float* Channel_Out;
	int* D;						// Num_Frames_OneTime_define֡���ݵ�������+У����,������CPU�ڴ���
	int ThreadPerBlock, NumBlock;
	int stopflag;
	float TimeGPU;
	LDPCCode* LDPC;


	cudaEvent_t GPU_start;			// GPU����ͳ�Ʋ���
	cudaEvent_t GPU_stop;
	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_stop);

	
	// ����ϵͳ�е�GPU����,��ָ��������һ��,ͬʱ�õ���GPU�����ܲ���
	cudaStatus = cudaGetDeviceCount(&Num_Device);
	if (cudaStatus != cudaSuccess)	// û��һ��������ڼ����GPU,���������в����޷�����
	{
		printf("There is no GPU beyond 1.0, exit!\n");
		//getch();
		exit(0);
	}
	else
	{
		cudaStatus = cudaGetDeviceProperties(&prop, Num_Device - 1);	// ѡ�����һ��GPU���ڼ���,ͬʱ����������ܲ���
		if (cudaStatus != cudaSuccess)	// û��һ��������ڼ����GPU,���������в����޷�����
		{
			printf("Cannot get device properties, exit!\n");
			//getch();
			exit(0);
		}
//		printf( "Clock rate:  %d\n", prop.clockRate );
//		printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
	}

	LDPC = (LDPCCode*)malloc(sizeof(LDPCCode));
	if (LDPC == NULL)
	{
		printf("Can not malloc LDPC in main on Host!\n");
		//getch();
		exit(0);
	}
	CodeWord = (int*)malloc(Num_Frames_OneTime * CW_Len * sizeof(int));
	if (CodeWord == NULL)
	{
		printf("Cannot malloc CodeWord in SNR_Simulation_GPU on host, exit!\n");
		//getch();
		exit(0);
	}	
	cudaStatus = cudaMalloc((void**)&CodeWord_GPU, Num_Frames_OneTime * CW_Len * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc CodeWord_GPU in SNR_Simulation_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	cudaStatus = cudaMalloc((void**)&Channel_Out_GPU, Num_Frames_OneTime * CW_Len * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		printf("Cannot malloc Channel_Out_GPU in SNR_Simulation_GPU on device, exit!\n");
		//getch();
		exit(0);
	}
	Channel_Out = (float*)malloc(Num_Frames_OneTime * CW_Len * sizeof(float));
	if (Channel_Out == NULL)
	{
		printf("Cannot malloc Channel_Out in SNR_Simulation_GPU on host, exit!\n");
		//getch();
		exit(0);
	}
	D = (int*)malloc((CW_Len + 1) * Num_Frames_OneTime * sizeof(int));
	if (D == NULL)
	{
		printf("Cannot malloc D in SNR_Simulation_GPU on host, exit!\n");
		//getch();
		exit(0);
	}
	while (1)
	{
		SIM->num_Frames += Num_Frames_OneTime;

		// ������������
		if (PN_Message == 0)	// ���汾����ȫ������
		{
			memset(CodeWord, 0, CW_Len * Num_Frames_OneTime * sizeof(int));
			cudaStatus = cudaMemset(CodeWord_GPU, 0, Num_Frames_OneTime * CW_Len * sizeof(int));
			if (cudaStatus != cudaSuccess)
			{
				printf("cudaMemset CodeWord_GPU cannot execute, exit!\n");
				//getch();
				exit(0);
			}
		}
		else if (PN_Message == 1)	// PN����,��Ҫ����
		{
		}

		// Num_Frames_OneTime_define֡�������ξ���AWGN�ŵ�,�õ���Ӧ���ŵ����.����,�����ֵ����������һ��.
		ThreadPerBlock = prop.maxThreadsPerBlock;
		NumBlock = (CW_Len % ThreadPerBlock == 0) ? CW_Len / ThreadPerBlock : CW_Len / ThreadPerBlock + 1;//��֤������	
		if (Add_noise == 1)
		{

			AWGNChannel_CPU(AWGN,Channel_Out,CodeWord);
			cudaMemcpy(Channel_Out_GPU, Channel_Out, Num_Frames_OneTime * CW_Len * sizeof(float), cudaMemcpyHostToDevice);
		}
		else BPSK << <NumBlock, ThreadPerBlock >> > (Channel_Out_GPU, CodeWord_GPU);
		
		LDPC_Decoder_GPU(D, Channel_Out_GPU, prop, Address_Variablenode, Weight_Checknode, Weight_Variablenode,LDPC);

		// ͳ��16֡�Ľ��
		stopflag = Statistic(SIM, CodeWord, D, LDPC);
		// if (SIM->num_Frames >= leastTestFrames)
		// {
		// 	exit(0);
		// }
		if (stopflag == 1)
		{
			break;
		}
		//cudaThreadSynchronize();
	}
	cudaEventRecord(GPU_stop, 0);
	cudaEventSynchronize(GPU_stop);
	cudaEventElapsedTime(&TimeGPU, GPU_start, GPU_stop);
	//printf( "Time of GPU for 1 iteration of LDPC code is:  %f us\n", TimeGPU);

	cudaEventDestroy(GPU_start);
	cudaEventDestroy(GPU_stop);

	free(LDPC);
	free(CodeWord);
	cudaFree(CodeWord_GPU);
	cudaFree(Channel_Out_GPU);
	free(D);
	free(Channel_Out);
}


/*
* ���������ʾ����
*/
void WriteLogo(AWGNChannel* AWGN, Simulation* SIM)
{

	/*�����������ӡ����Ļ��*/
	printf("*******************Binary LDPC Simulation*******************\n");
	printf("*Author: Lv Yanchen                         Date:2020/9/28\n\n");
	printf("* Message bits' length of LDPC is %d\n", msgLen);
	printf("* Parity bits' length of LDPC is %d\n", parLen);
	printf("* CodeWord length of LDPC is %d\n", CW_Len);
	printf("* H's row is divided into %d blocks, and column divided into %d blocks. Dimension Z is %d\n", J, L, Z);
	printf("* The encoding rate for current LDPC is %f\n", rate);
	if (PN_Message == 0)
	{
		printf("* Information bits are zero sequence, encoder is no need here.\n");
	}
	else if (PN_Message == 1)
	{
		printf("* Information bits are generated by PN sequence.\n");
	}
	printf("* Maximum iterations for LDPC_decoder is %d\n", maxIT);

	if (decoder_method == 0)
	{
		printf("* LDPC decoder use normalized min-sum algorithm!\n");
		//printf("* Optimal factor opt for R in NMS is: %f\n", opt_R);
	}

	if (Add_noise == 0)
	{
		printf("* Not add white gaussin noise on the symbol.\n");
	}
	else if (Add_noise == 1)
	{
		printf("* Add white gaussin noise on the symbol.\n");
	}
	printf("* Initial seeds for each SNR are %d, %d, %d.\n", AWGN->seed[0], AWGN->seed[1], AWGN->seed[2]);

	if (snrtype == 0)
	{
		printf("* The type of SNR is Eb/No\n");
	}
	else if (snrtype == 1)
	{
		printf("* The type of SNR is Es/No\n");
	}
	printf("* Simulation SNR(SNR_start SNR_stop SNR_step) are: %.2f, %.2f, %.2f\n", startSNR, stopSNR, stepSNR);

	printf("* Least error frames to exit the simulation for each SNR is %d.\n", leastErrorFrames);
	printf("* Least test frames to exit the simulation for each SNR is %d.\n", leastTestFrames);
	printf("* Display step is %d.\n", displayStep);

	if (CPU_GPU == 0)
	{
		printf("* Simulation is on CPU.\n");
	}
	else if (CPU_GPU == 1)
	{
		printf("* Simulation is on GPU.\n");
	}

	printf("* %d frames are simulated simultaneously.\n", Num_Frames_OneTime);

	printf("***************************************************************************\n\n\n");
	printf(" SNR   %5s   %5s   %7s    %7s     %7s  %7s   %7s\n", "NTF", "NEF", "FER", "BER", "AverIT", "FER_F", "FER_A");	
}

/*
* ͳ�ƺ�����ͳ�Ʒ�����
*/
int Statistic(Simulation* SIM, int* CodeWord_Frames, int* D,LDPCCode *LDPC)
{
	int index0, index1, Length;
	int Error_msgBit[Num_Frames_OneTime];	// ???????��???????��???????
	Length = (Message_CW == 0) ? msgLen : CW_Len;

	memset(Error_msgBit, 0, Num_Frames_OneTime * sizeof(int));
	for (index0 = 0; index0 < Num_Frames_OneTime; index0++)
	{
		for (index1 = 0; index1 < Length; index1++)
		{
			Error_msgBit[index0] = (D[index1 * Num_Frames_OneTime + index0] != CodeWord_Frames[index1 * Num_Frames_OneTime + index0]) ? Error_msgBit[index0] + 1 : Error_msgBit[index0];
		}
		SIM->num_Error_Bits += Error_msgBit[index0];
		SIM->num_Error_Frames = (Error_msgBit[index0] != 0 || D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Error_Frames + 1 : SIM->num_Error_Frames;
		SIM->num_Alarm_Frames = (Error_msgBit[index0] == 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 0) ? SIM->num_Alarm_Frames + 1 : SIM->num_Alarm_Frames;
		SIM->num_False_Frames = (Error_msgBit[index0] != 0 && D[index0 + CW_Len * Num_Frames_OneTime] == 1) ? SIM->num_False_Frames + 1 : SIM->num_False_Frames;
		SIM->Total_Iteration += LDPC->iteraTime;
	}
	if (SIM->num_Frames % displayStep == 0)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(Length);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4e %6.4e\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->FER_False, SIM->FER_Alarm);
	}

	if (SIM->num_Error_Frames >= leastErrorFrames && SIM->num_Frames >= leastTestFrames)
	{
		SIM->BER = ((double)SIM->num_Error_Bits / (double)(SIM->num_Frames)) / (double)(Length);
		SIM->FER = (double)SIM->num_Error_Frames / (double)SIM->num_Frames;
		SIM->AverageIT = (double)SIM->Total_Iteration / (double)SIM->num_Frames;
		SIM->FER_Alarm = (double)SIM->num_Alarm_Frames / (double)SIM->num_Frames;
		SIM->FER_False = (double)SIM->num_False_Frames / (double)SIM->num_Frames;
		printf(" %.1f %8d  %4d  %6.4e  %6.4e  %.2f  %6.4e %6.4e\n", SIM->SNR, SIM->num_Frames, SIM->num_Error_Frames, SIM->FER, SIM->BER, SIM->AverageIT, SIM->FER_False, SIM->FER_Alarm);
		return 1;
	}
	return 0;
}

/*
H:У�����
Weight_Checknode:��˳���¼ÿ��У��ڵ�����������һλΪ�������
Weight_Variablenode:��˳���¼ÿ�������ڵ�����������һλΪ�������
*/
void Get_H(int* H, int* Weight_Checknode, int* Weight_Variablenode)
{
	int index0, index1;
	char temp[100];
	char file[100];
	FILE* fp_H;
	strcpy(file, "J4");
	//_itoa(J, temp, 10);
	//strcat(file, temp);
	//_itoa(L, temp, 10);
	strcat(file, "_L24");
	//strcat(file, temp);
	//_itoa(Z, temp, 10);
	strcat(file, "_Z96");
	//strcat(file, temp);
	strcat(file, "_BlockH.txt");
	if (NULL == (fp_H = fopen(file, "r")))
	{
		printf("can not open file: %s\n", file);
		getchar();
		exit(0);
	}

	for (index0 = 0; index0 < L *J; index0++)
	{
		fscanf(fp_H, "%d", &index1);
		*(H + index0) = index1;
	}
	fclose(fp_H);

	for (index0 = 0; index0 < J; index0++)
	{
		for (index1 = 0; index1 < L; index1++)
		{
			Weight_Checknode[index0] = (H[index0 * L + index1] != -1) ? Weight_Checknode[index0] + 1 : Weight_Checknode[index0];//????-1??????1???????��?????(��????)
		}
		// ѡ���������
		Weight_Checknode[J] = (Weight_Checknode[index0] > Weight_Checknode[J]) ? Weight_Checknode[index0] : Weight_Checknode[J];//��????????????
	}


	for (index0 = 0; index0 < L; index0++)
	{
		for (index1 = 0; index1 < J; index1++)
		{
			Weight_Variablenode[index0] = (H[index1 * L + index0] != -1) ? Weight_Variablenode[index0] + 1 : Weight_Variablenode[index0];//????-1??????1???????��?????(???????)
		}
		// ѡ���������
		Weight_Variablenode[L] = (Weight_Variablenode[index0] > Weight_Variablenode[L]) ? Weight_Variablenode[index0] : Weight_Variablenode[L];//???????????????
	}

	if (Weight_Checknode[J] > maxWeight_checknode || Weight_Checknode[J] < minWeight_checknode)//?????????��
	{
		printf("You must input a LDPC code with Weight_Checknode in [%d, %d], exit!\n", minWeight_checknode, maxWeight_checknode);
		//getch();
		exit(0);
	}
	if (Weight_Variablenode[L] > maxWeight_variablenode || Weight_Variablenode[L] < minWeight_variablenode)
	{
		printf("You must input a LDPC code with Weight_variablenode in [%d, %d], exit!\n", minWeight_variablenode, maxWeight_variablenode);
		//getch();
		exit(0);
	}
}

/*
* H:У�����
* Weight_Checknode:У��ڵ�����
* Weight_Variablenode:�����ڵ�����
* Address_Variablenode:ÿ�������ڵ�����ӦУ��ڵ��memory_rq�ĵ�ַ
* У��ڵ㲻��Ҫ����ΪУ��ڵ��Ӧ�ľ���ÿһ��memory_rq����ַ������һ���
*/
void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode)
{
	int index0, index1, index2, index3, index4, position;
	for (index0 = 0; index0 < L; index0++)		// index0Ϊ��ǰ������
	{
		index2 = 0;
		for (index1 = 0; index1 < J; index1++)	// index1Ϊ��ǰ������
		{
			if (H[index1 * L + index0] != -1)
			{
				position = 0;	// �����ڵ�������У��ڵ���,�ñ����ڵ�����λ��(ÿһ��1��Ӧ��λ��)
				for (index3 = 0; index3 < index0; index3++)
				{
					position = (H[index1 * L + index3] != -1) ? position + 1 : position;//ÿһ�еڼ����ȫ������
				}
				for (index3 = 0; index3 < Z; index3++)//index3(��)��ÿ��һ����ͳ����һ�������ڵ�����ӹ�ϵ
				{
					index4 = (((Z - H[index1 * L + index0]) % Z + index3) >= Z) ? (Z - H[index1 * L + index0]) % Z + index3 - Z : index3;//zά�����ÿһ�е�1�ڵڼ��У��������ʽ�ӣ�
					Address_Variablenode[(index0 * Z + index3) * Weight_Variablenode[L] + index2] = (index1 * Z + index4) * Weight_Checknode[J] + position;//��¼ÿ�������ڵ㣨ÿһ�У�����Щ�����ӣ����Ϊ���к��У�
				}
				index2++;//��һ�У��飩
			}
		}
	}
}