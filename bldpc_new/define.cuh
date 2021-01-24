#ifndef _DEFINE_H_
#define _DEFINE_H_

#include <stdio.h>

#define PI (3.1415926)

#define	J 4	                    //行						
#define	L 24                    //列
#define	Z 96                    //块维度
#define	CW_Len 2304             //序列总长度
#define	msgLen 1920             //信息位长度
#define	parLen 384              //校验位长度

#define maxIT 50                //最大迭代次数

#define leastErrorFrames 50     //最少得仿真出多少错误帧
#define leastTestFrames 10000   //最少仿真帧数

#define Num_Frames_OneTime 1

#endif