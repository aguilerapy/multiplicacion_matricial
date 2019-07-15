#include "matmat.h"
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <x86intrin.h>

// double** mat a (mat N*N)
// double** mat b (mat N*N)

int 
main(int argc, char** argv) {
	// mpi funneled initialization
	int provided, pid, nump, namelen;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &nump);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Get_processor_name(hostname, &namelen);

	// time struct
	struct timespec ts_start;
	struct timespec ts_stop;

	// mat initialization
	double** mat_a;
	mat_a = create_mat();
	
	double** mat_b;
	mat_b = create_mat();

	double** mat_c;
	mat_c = create_mat();

	// send and receive mat_a
	if (pid == 0) {
		gen_mat(mat_a);
		gen_mat(mat_b);

		// timer start
		clock_gettime(CLOCK_MONOTONIC, &ts_start);

		for (int process = 1; process < nump; ++process) {
			// parts of mat_a
			for (int i = process; i < N; i+=nump) {
				MPI_Send (mat_a[i], N, MPI_DOUBLE, process, 0, MPI_COMM_WORLD);
			}
		}

	} else {
		// parts of mat_a
		for (int i = pid; i < N; i+=nump) {
			MPI_Recv (mat_a[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	// bcast all mat_b
	for (int i = 0; i < N; ++i) {
		MPI_Bcast(mat_b[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	// avx variables
	__m256d ma1;
	__m256d ma2;
	__m256d ma3;
	__m256d ma4;
	__m256d mb1;
	__m256d mb2;	
	__m256d mc1;
	__m256d mc2;

	// hotspot
  	#pragma omp parallel num_threads(8) default(none) shared(mat_c,mat_b,mat_a,nump,pid) private(ma1,ma2,ma3,ma4,mb1,mb2,mc1,mc2) 
	{
    	#pragma omp for schedule(static)
			for(int i = pid; i < N; i+=(2*nump))
			{	
				for (int k = 0; k < N; k+=2)
				{
					ma1 = _mm256_broadcast_sd(&mat_a[i][k]); 
					ma2 = _mm256_broadcast_sd(&mat_a[i][k+1]); 
					ma3 = _mm256_broadcast_sd(&mat_a[i+nump][k]); 
					ma4 = _mm256_broadcast_sd(&mat_a[i+nump][k+1]);							
					for (int j = 0; j < N; j+=4)
					{
						mc1 = _mm256_loadu_pd(&mat_c[i][j]); 
						mc2 = _mm256_loadu_pd(&mat_c[i+nump][j]);
							
						mb1 = _mm256_loadu_pd(&mat_b[k][j]);
						mb2 = _mm256_loadu_pd(&mat_b[k+1][j]);

						mc1 = _mm256_add_pd(mc1, _mm256_add_pd(_mm256_mul_pd(ma1, mb1), _mm256_mul_pd(ma2, mb2))); 
						mc2 = _mm256_add_pd(mc2, _mm256_add_pd(_mm256_mul_pd(ma3, mb1), _mm256_mul_pd(ma4, mb2)));

						_mm256_storeu_pd(&mat_c[i][j], mc1);
						_mm256_storeu_pd(&mat_c[i+nump][j], mc2);
					}
				}
			}
	}

	// slaves send results to master
	if (pid != 0) {
		for (int i = pid; i < N; i+=nump) {
			MPI_Send (mat_c[i], N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}

	} else  {

		// master receive results
		for (int process = 1; process < nump; ++process) {
			for (int i = process; i < N; i+=nump) {
				MPI_Recv (mat_c[i], N, MPI_DOUBLE, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		//timer stop
		clock_gettime(CLOCK_MONOTONIC, &ts_stop);		

		// verification
		double sum = 0.0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				sum += mat_c[i][j];
			}
		}
		
		double start = (double)ts_start.tv_sec + (double)ts_start.tv_nsec/1000000000.0;
		double stop = (double)ts_stop.tv_sec + (double)ts_stop.tv_nsec/1000000000.0;
		double elapsed = (stop - start);		

		// display results
		printf ("sum = %f time = %f\n", sum, elapsed);

	}

	// free resource
	delete_mat(mat_a);
	delete_mat(mat_b);
	delete_mat(mat_c);

	MPI_Finalize();
	
	return 0;
}
