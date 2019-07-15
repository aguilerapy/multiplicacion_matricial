#ifndef __MATMAT_H__
#define __MATMAT_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 512
//#define N 1024
//#define N 2048
//#define N 4096

double
random_number(void) {
	int value;
	int sign;

	// srand(time(NULL));
	value = rand() % 100;
	sign = rand() % 2;

	if(sign == 0) {
		value *= -1;
	}

	return ((double) value);
}

void
gen_mat(double** mat) {
	int i, j;
	for(i = 0; i < N; ++i) {
		for(j = 0; j < N; ++j) {
			mat[i][j] = random_number();
			// mat[i][j] = 1.0;
		}
	}
}

void 
gen_vec(double* vec) {
	int i;
	for(i = 0; i < (N*N); ++i) {
		vec[i] = random_number();
		// vec[i] = 1.0;
	}
}

double**
create_mat() {
	double** mat;
	
	if ((mat = (double**) malloc(sizeof(double*) * N)) == NULL) {
		fprintf(stderr, "Can't allocate memory...\n");
		exit(1);
	}

	int i;
	for (i = 0; i < N; ++i) {
		if ((mat[i] = (double*) malloc(sizeof(double) * N)) == NULL) {
			fprintf(stderr, "Can't allocate memory...\n");
			exit(1);
		}
		memset(mat[i], 0, sizeof (double) * N);
	}

	return mat;
}

double*
create_vec() {
	double* vec;

	if ((vec = (double*) malloc(sizeof(double*) * (N*N))) == NULL) {
		fprintf(stderr, "Can't allocate memory...\n");
		exit(1);
	}
	memset(vec, 0, sizeof (double) * N);

	return vec;
}

void 
delete_mat(double** mat) {
	int i;
	for (i = 0; i < N; ++i) {
		free(mat[i]);
	}
	free(mat);
}

void 
delete_vec(double* vec) {
	free(vec);
}

#endif

