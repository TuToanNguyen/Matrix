// nhan2matran.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
// Matrix Mul without omp.cpp : Defines the entry point for the console application.
//
// Nhan hai ma tran khong su dung OpenMP
#include "StdAfx.h"
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <float.h>
#include <iostream>
//#define N 800 // Kich thuoc ma tran NxN
using namespace std;
int main()
{
	cout << "Nhap cap cua ma tran n: ";
	int N;
	cin >> N;
	int **a, **b, **c;
	clock_t start, end; //Bien thoi gian
	int i, j, k;
	double dif; // Bien qui doi thoi gian ra giay
				//Cap phat bo nho cho ma tran A
	a = (int **)malloc(10 * N);
	for (i = 0; i<N; i++)
	{
		a[i] = (int *)malloc(10 * N);
	}
	//Cap phat bo nho cho ma tran B
	b = (int **)malloc(10 * N);
	for (i = 0; i<N; i++)
	{
		b[i] = (int *)malloc(10 * N);
	}
	//Cap phat bo nho cho ma tran tong
	c = (int **)malloc(10 * N);
	for (i = 0; i< N; i++)
	{
		c[i] = (int *)malloc(10 * N);
	}
	printf("Khoi tao ma tran A...\n");
	start = clock(); //Bat dau dem thoi gian
	//Khoi tao ma tran A
	for (i = 0; i<N; i++)
	{
		for (j = 0; j<N; j++)
		{
				a[i][j] = i + j;
				cout << a[i][j] << "\t";
		}
		cout << endl;
	}
	printf("Khoi tao ma tran B...\n");
	//Khoi tao ma tran B
	for (i = 0; i<N; i++)
	{
		for (j = 0; j<N; j++)
		{
			b[i][j] = i * j;
			cout << b[i][j] << "\t";
		}
		cout << endl;
	}
	//Khoi tao ma tran C
	for (i = 0; i<N; i++)
	{
		for (j = 0; j< N; j++)
		{
			c[i][j] = 0;
		}
	}
	printf("Nhan hai ma tran.....\n");
	for (i = 0; i<N; i++)
	{
		for (j = 0; j<N; j++)
		{
			for (k = 0; k<N; k++)
			{
				c[i][j] += a[i][k] * b[k][j];
			}
			cout << c[i][j] << "\t";
		}
		cout << endl;
	}
	end = clock(); //Thoi gian ket thuc
	dif = ((double)(end - start)) / CLOCKS_PER_SEC; //Qui doi thoi gian ra giay
	printf("Xong. Thoi gian thuc hien het: %f giay.\n", dif);
	// Tien hanh giai phong bo nho cho cac ma tran da cap phat
	for (i = 0; i<N; i++)
	{
		free(a[i]);
	}
	free(a);
	for (i = 0; i<N; i++)
	{
		free(b[i]);
	}
	free(b);
	for (i = 0; i<N; i++)
	{
		free(c[i]);
	}
	free(c);
	_getch();
}

