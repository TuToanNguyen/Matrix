// Nhan hai ma tran su dung OpenMP
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <omp.h>
#include <iostream>
//#define N 200// Kich thuoc ma tran NxN
using namespace std;
int main(int argc, char *argv[])
{
	cout << "Nhap cap cua ma tran n: ";
	int N;
	cin >> N;

	int **a, **b, **c; // Khai báo mảng 2 chiều so nguyen
	int tid, nthreads, chunk = 10; //nthreads: bien de luu so threads duoc su dung;tid: Bien de luu so thu tu cua thread
								   //chunk = 10: Sử dụng để chia trong vong lap For
	double dif; //Bien luu thoi gian tinh toan
	int i, j, k; // Bien su dung cho vong lap
				 /* Cap phat bo nho cho ma tran A */
	a = (int **)malloc(10 * N);
	for (i = 0; i<N; i++)
	{
		a[i] = (int *)malloc(10 * N);
	}
	/* Cap phat bo nho cho ma tran B */
	b = (int **)malloc(10 * N);
	for (i = 0; i<N; i++)
	{
		b[i] = (int *)malloc(10 * N);
	}
	/* Cap phat bo nho cho ma tran C */
	c = (int **)malloc(10 * N);
	for (i = 0; i< N; i++)
	{
		c[i] = (int *)malloc(10 * N);
	}
	printf("Khoi tao ma tran...\n");
	double start = omp_get_wtime(); //Bat dau dem thoi gian

									/*** Tao mot vung song song voi cac bien chia se giua cac thread gom: a,b,c nthread, chunks va cac bien rieng duoc su dung cho cac thread la i,j,k ***/
#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
	{
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			nthreads = omp_get_num_threads();// Lay so thread duoc su dung
			printf("Bat dau nhan ma tran voi so thread la: %d threads\n", nthreads);
		}
		//Khoi tao ma tran A
		/* Khai bao mot vong lap duoc thuc hien song song giua cac thread
		voi lich trinh Tinh va kich thuoc moi doan la 10*/
#pragma omp single

		printf("Khoi tao ma tran A...\n");

		
#pragma omp for schedule (static, chunk)
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<N; j++)
			{
				a[i][j] = i + j;
				cout << a[i][j] << "\t";
			}
			cout << endl;
		}
		//Khoi tao ma tran B
		/* Khai bao mot vong lap duoc thuc hien song song giua cac thread
		voi lich trinh Tinh va kich thuoc moi doan la 10*/
#pragma omp single

		printf("Khoi tao ma tran B...\n");
#pragma omp for schedule (static, chunk)
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
		/* Khai bao mot vong lap duoc thuc hien song song giua cac thread
		voi lich trinh Tinh va kich thuoc moi doan la 10*/
#pragma omp single

		printf("Nhan hai ma tran.....\n");
#pragma omp for schedule (static, chunk)
		for (i = 0; i<N; i++)
		{
			for (j = 0; j<N; j++)
			{
				c[i][j] = 0;
			}
			
		}
	
		printf("Thread %d dang tien hanh nhan ma tran...\n", tid);

#pragma omp for schedule (static, chunk)
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
	} /*****Ket thuc vung song song*****/
	double end = omp_get_wtime(); //Thoi gian ket thuc
	dif = end - start; //Khoang thoi gian thuc hien
	printf("Xong.Thoi gian thuc hien la: %f giay.\n", dif);
	/*Giai phong bo nho*/
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
