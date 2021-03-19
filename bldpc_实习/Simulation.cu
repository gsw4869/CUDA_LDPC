#include "Simulation.cuh"
#include "LDPC_Encoder.cuh"
#include "LDPC_Decoder.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*
* 仿真函数
* AWGN:AWGNChannel类变量，包含噪声种子等
* 
*/
void Simulation_GPU(AWGNChannel* AWGN, float* sigma_GPU, Simulation* SIM, int* Address_Variablenode, int* Weight_Checknode, int* Weight_Variablenode)
{
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	int Num_Device;
	int* CodeWord;
	int* CodeWord_GPU;			// 编码所得码字序列,Num_Frames_OneTime_define帧,分配在GPU内存中
	float* Channel_Out_GPU;		// Num_Frames_OneTime_define帧数据经过AWGN信道后的结果,分配在GPU内存中
	float* Channel_Out;
	int* D;						// Num_Frames_OneTime_define帧数据的译码结果+校验结果,分配在CPU内存中
	int ThreadPerBlock, NumBlock;
	int stopflag;
	float TimeGPU;
	LDPCCode* LDPC;


	cudaEvent_t GPU_start;			// GPU速率统计参数
	cudaEvent_t GPU_stop;
	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_stop);

	
	// 查找系统中的GPU个数,并指定采用那一块,同时得到该GPU的性能参数
	cudaStatus = cudaGetDeviceCount(&Num_Device);
	if (cudaStatus != cudaSuccess)	// 没有一块可以用于计算的GPU,则下列所有步骤无法进行
	{
		printf("There is no GPU beyond 1.0, exit!\n");
		//getch();
		exit(0);
	}
	else
	{
		cudaStatus = cudaGetDeviceProperties(&prop, Num_Device - 1);	// 选择最后一块GPU用于计算,同时获得它的性能参数
		if (cudaStatus != cudaSuccess)	// 没有一块可以用于计算的GPU,则下列所有步骤无法进行
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

		// 产生码字序列
		if (PN_Message == 0)	// 本版本均用全零序列
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
		else if (PN_Message == 1)	// PN序列,需要编码
		{
		}

		// Num_Frames_OneTime_define帧数据依次经过AWGN信道,得到相应的信道输出.其中,各码字的输出穿插在一起.
		ThreadPerBlock = prop.maxThreadsPerBlock;
		NumBlock = (CW_Len % ThreadPerBlock == 0) ? CW_Len / ThreadPerBlock : CW_Len / ThreadPerBlock + 1;//保证够处理	
		if (Add_noise == 1)
		{

			AWGNChannel_CPU(AWGN,Channel_Out,CodeWord);
			cudaMemcpy(Channel_Out_GPU, Channel_Out, Num_Frames_OneTime * CW_Len * sizeof(float), cudaMemcpyHostToDevice);
		}
		else BPSK << <NumBlock, ThreadPerBlock >> > (Channel_Out_GPU, CodeWord_GPU);
		
		LDPC_Decoder_GPU(D, Channel_Out_GPU, prop, Address_Variablenode, Weight_Checknode, Weight_Variablenode,LDPC);

		// 统计16帧的结果
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
* 仿真参数显示函数
*/
void WriteLogo(AWGNChannel* AWGN, Simulation* SIM)
{

	/*将仿真参数打印到屏幕上*/
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
* 统计函数，统计仿真结果
*/
int Statistic(Simulation* SIM, int* CodeWord_Frames, int* D,LDPCCode *LDPC)
{
	int index0, index1, Length;
	int Error_msgBit[Num_Frames_OneTime];	// ???????д???????λ???????
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
H:校验矩阵
Weight_Checknode:按顺序记录每个校验节点的重量，最后一位为最大重量
Weight_Variablenode:按顺序记录每个变量节点的重量，最后一位为最大重量
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
			Weight_Checknode[index0] = (H[index0 * L + index1] != -1) ? Weight_Checknode[index0] + 1 : Weight_Checknode[index0];//????-1??????1???????е?????(У????)
		}
		// 选择最大重量
		Weight_Checknode[J] = (Weight_Checknode[index0] > Weight_Checknode[J]) ? Weight_Checknode[index0] : Weight_Checknode[J];//У????????????
	}


	for (index0 = 0; index0 < L; index0++)
	{
		for (index1 = 0; index1 < J; index1++)
		{
			Weight_Variablenode[index0] = (H[index1 * L + index0] != -1) ? Weight_Variablenode[index0] + 1 : Weight_Variablenode[index0];//????-1??????1???????е?????(???????)
		}
		// 选择最大重量
		Weight_Variablenode[L] = (Weight_Variablenode[index0] > Weight_Variablenode[L]) ? Weight_Variablenode[index0] : Weight_Variablenode[L];//???????????????
	}

	if (Weight_Checknode[J] > maxWeight_checknode || Weight_Checknode[J] < minWeight_checknode)//?????????Χ
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
* H:校验矩阵
* Weight_Checknode:校验节点重量
* Weight_Variablenode:变量节点重量
* Address_Variablenode:每个变量节点所对应校验节点的memory_rq的地址
* 校验节点不需要是因为校验节点对应的就是每一行memory_rq，地址是连在一起的
*/
void Transform_H(int* H, int* Weight_Checknode, int* Weight_Variablenode, int* Address_Variablenode)
{
	int index0, index1, index2, index3, index4, position;
	for (index0 = 0; index0 < L; index0++)		// index0为当前所在列
	{
		index2 = 0;
		for (index1 = 0; index1 < J; index1++)	// index1为当前所在行
		{
			if (H[index1 * L + index0] != -1)
			{
				position = 0;	// 变量节点相连的校验节点中,该变量节点的相对位置(每一行1对应的位置)
				for (index3 = 0; index3 < index0; index3++)
				{
					position = (H[index1 * L + index3] != -1) ? position + 1 : position;//每一行第几块非全零矩阵块
				}
				for (index3 = 0; index3 < Z; index3++)//index3(列)，每加一就是统计下一个变量节点的连接关系
				{
					index4 = (((Z - H[index1 * L + index0]) % Z + index3) >= Z) ? (Z - H[index1 * L + index0]) % Z + index3 - Z : index3;//z维方块里，每一列的1在第几行（结合下面式子）
					Address_Variablenode[(index0 * Z + index3) * Weight_Variablenode[L] + index2] = (index1 * Z + index4) * Weight_Checknode[J] + position;//记录每个变量节点（每一列）和哪些点连接（序号为先行后列）
				}
				index2++;//下一行（块）
			}
		}
	}
}