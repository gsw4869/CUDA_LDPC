#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "define.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LDPC_Decoder.cuh"
#include "struct.cuh"

int main()
{
	
	CN* Checknode;			// LDPC码各分块中校验节点的重量
	VN* Variablenode;		// LDPC码各分块中变量节点的重量
	Get_H(Variablenode,Checknode);
	return 0;
}