/*
 * @Author: your name
 * @Date: 2021-04-09 16:52:00
 * @LastEditTime: 2021-04-09 17:18:09
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /myNBLDPC/src/GF.cpp
 */
#include "GF.cuh"
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>
using namespace std;

unsigned **TableAdd;
unsigned **TableMultiply;
unsigned *TableInverse;

unsigned **malloc_2(int xDim, int yDim)
{
	unsigned **a = (unsigned **)malloc(xDim * sizeof(unsigned *));
	a[0] = (unsigned *)malloc(xDim * yDim * sizeof(unsigned));
	memset(a[0], 0, xDim * yDim * sizeof(unsigned));
	for (int i = 1; i < xDim; i++)
	{
		a[i] = a[i - 1] + yDim;
	}
	assert(a != NULL);
	return a;
}

float **malloc_2_float(int xDim, int yDim)
{
	float **a = (float **)malloc(xDim * sizeof(float *));
	a[0] = (float *)malloc(xDim * yDim * sizeof(float));
	memset(a[0], 0, xDim * yDim * sizeof(float));
	for (int i = 1; i < xDim; i++)
	{
		a[i] = a[i - 1] + yDim;
	}
	assert(a != NULL);
	return a;
}

int GFAdd(int ele1, int ele2)
{
	return ele1 ^ ele2;
}

int GFMultiply(int ele1, int ele2)
{
	return TableMultiply[ele1][ele2];
}

int GFInverse(int ele)
{
	if (ele == 0)
	{
		printf("Div 0 Error!\n");
		exit(-1);
	}
	return TableInverse[ele];
}

bool GFInitial(int GFq)
{
	// calculate order
	int q = GFq;
	// allocate memory space
	TableAdd = malloc_2(q, q);
	TableMultiply = malloc_2(q, q);
	TableInverse = new unsigned[q];

	// read profile
	stringstream ss;
	ss << q << ".txt";
	//Arithmetic Table
	string ArithTableFileName = "./GF/Arith.Table.GF.";
	ArithTableFileName += ss.str();
	ifstream ArithFin(ArithTableFileName);
	if (!ArithFin.is_open())
	{
		cerr << "Cannot open " << ArithTableFileName << endl;
		exit(-1);
	}
	string rub;
	getline(ArithFin, rub);
	//	cout << "Read Arithmetic Table File: " << rub << "..." << endl;
	ArithFin >> rub >> rub;
	for (int i = 0; i < q; i++)
	{
		for (int j = 0; j < q; j++)
		{
			ArithFin >> TableMultiply[i][j];
		}
	}
	ArithFin >> rub >> rub;
	for (int i = 0; i < q; i++)
	{
		for (int j = 0; j < q; j++)
		{
			ArithFin >> TableAdd[i][j];
		}
	}
	ArithFin >> rub >> rub;
	for (int i = 0; i < q; i++)
	{
		ArithFin >> TableInverse[i];
	}
	ArithFin.close();
	//	cout << "done." << endl;

	return true;
}
