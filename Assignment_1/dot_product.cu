#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
//#include <cuda_runtime_api.h>
#include "dot_product.h"

int main(int argc ,char* argv[]) {
	
	FILE *data;
	FILE *vector;
	size_t size;
	size_t sizeV;
	
	/* Initialize rows, cols, CUDA devices and threads from the user */
	unsigned int rows=atoi(argv[3]);
	unsigned int cols=atoi(argv[4]);					  
	int CUDA_DEVICE = atoi(argv[5]);
	int THREADS = atoi(argv[6]);
	
	printf("Rows= %d\n,Cols = %d\n,CUDA_DEVICE= %d\n, THREADS =%d \n",rows,cols,CUDA_DEVICE,THREADS);
	cudaError err = cudaSetDevice(CUDA_DEVICE);

	if(err != cudaSuccess) { printf("Error setting CUDA DEVICE\n"); exit(EXIT_FAILURE); }

	/*Host variable declaration */

	//int THREADS = 32;				
	int BLOCKS;
	float* host_results = (float*) malloc(rows * sizeof(float)); 
	struct timeval starttime, endtime;
	clock_t start, end;
	float seconds = 0;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	
	float  *dev_dataT;
	float *dev_dataV;
	float *results;
        //size_t len = 0;
	float arr[rows][cols];
	float var ;
	int vrow =1;

	start = clock();

	/* Validation to check if the data file is readable */
	
	data = fopen(argv[1], "r");
	vector = fopen(argv[2],"r");
	
	if (data == NULL)
	{
    		printf("Cannot Open the data ");
		return 0;
	}
	if (vector == NULL)
	{
    		printf("Cannot Open the vector");
		return 0;
	}
	
	size = (size_t)((size_t)rows * (size_t)cols);
	sizeV = (size_t)((size_t)vrow*(size_t)cols);

	printf("Size of the data = %lu\n",size);
	printf("Size of the vector = %lu\n",sizeV);

	fflush(stdout);

	float *dataT = (float*)malloc((size)*sizeof(float));
	float *dataV = (float*)malloc((sizeV)*sizeof(float));

	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
	}
	if(dataV == NULL) {
	        printf("ERROR: Memory for vector not allocated.\n");
	}
	
        gettimeofday(&starttime, NULL);
	int j = 0;

    /* Transfer the Data from the file to CPU Memory */
	

        for (i =0; i< rows;i++){
		for(j=0; j<cols ; j++){
			fscanf(data,"%f",&var);
                        arr[i][j]=var;
		}
	}
	for (i =0;i<cols;i++){
		for(j= 0; j<rows; j++){
			dataT[rows*i+j]= arr[j][i];
	}
	}		

		for (j=0;j<cols;j++){
			fscanf(vector,"%f",&dataV[j]);
		}
   
	fclose(data);
	fclose(vector);
		printf("Read Data\n");
        fflush(stdout);

        gettimeofday(&endtime, NULL);
        seconds+=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);

        printf("time to read data = %f\n", seconds);

	/* Allocate the Memory in the GPU for data */

        gettimeofday(&starttime, NULL);
	err = cudaMalloc((float**) &dev_dataT, (size_t) size * (size_t) sizeof(float));
	if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for data =%f\n", seconds);

        gettimeofday(&starttime, NULL);


	/* Allocate the memory in the GPU for vector */
	
        err = cudaMalloc((float**) &dev_dataV, sizeV * sizeof(float));
       if(err != cudaSuccess) { printf("Error mallocing data on GPU device\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time for cudamalloc for vector =%f\n", seconds);

        gettimeofday(&starttime, NULL);
	
	err = cudaMalloc((float**) &results, rows * sizeof(float) );
	if(err != cudaSuccess) { printf("Error mallocing results on GPU device\n"); }
        gettimeofday(&endtime, NULL); 
seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for cudamalloc for result =%f\n", seconds);

	/*Copy the data to GPU */
	
	
        gettimeofday(&starttime, NULL);
	err = cudaMemcpy(dev_dataT, dataT, (size_t)size *sizeof(float), cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time to copy  data to GPU=%f\n", seconds);

	
	gettimeofday(&starttime, NULL);
        err = cudaMemcpy(dev_dataV, dataV, sizeV*sizeof(float), cudaMemcpyHostToDevice);
        if(err != cudaSuccess) { printf("Error copying data to GPU\n"); }
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
        printf("time to copy vector data to GPU=%f\n", seconds);

	jobs = rows;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

        gettimeofday(&starttime, NULL);

	/* Calling the kernel function */
	
	kernel<<<BLOCKS,THREADS>>>(rows,cols,dev_dataT,	dev_dataV, results);
        gettimeofday(&endtime, NULL); seconds=((double)endtime.tv_sec+(double)endtime.tv_usec/1000000)-((double)starttime.tv_sec+(double)starttime.tv_usec/1000000);
	printf("time for kernel=%f\n", seconds);
		
	/* Copy the results back in host */
	
	cudaMemcpy(host_results,results,rows * sizeof(float),cudaMemcpyDeviceToHost);
	
	printf("Output of dot product is \n");
	printf("\n");
	
	for(int k = 0; k < jobs; k++) {
		printf("%f ", host_results[k]);
	}
	printf("\n");

	cudaFree( dev_dataT );
	cudaFree( results );

	end = clock();
	seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Total time = %f\n", seconds);

	return 0;

}