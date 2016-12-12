/*
 * Created by Harshavardhan Patil on 9/30/16.
 *
 *
 *  Matrix Normalization using CUDA :
 *  - The generated input values are stored by inverting the matrix. i.e All the attributes of a column which needs to be normalized
 *    are stored as elements of a row. So that while normalizing the threads in a block will access nearby elements which will
 *    optimize the code
 *  - The number of threads is hard coded to 16. This value is finalized after multiple iterations with values (32, 64, 128, 256, 512).
 *    This value is used along with the size of the matrix to calculate number of blocks.
 *  - Each block is assigned to calculate sum and squares of one row.
 *    In a block each thread will read one element and put it in a shared memory. Once this is done, partial sum is calculated and
 *    each block stores it's partial sum in the global memory area using it's block Id as index.
 *  - Once partial sum calculation is done by all the blocks, another kernal function is launched with this partial
 *    sum as the input in a single block. This block calculates the final sum and squares.
 *  - Partial sum using reduction works only if the number of elements passed to the block is a power of 2.
 *    So to avoid wrong calculation when number of blocks is not power of 2 (in calculateFinalSum method).
 *    The input array argument lentgh is set to the nearest power of 2 for the number of blocks and value 0 is set to those indices
 *    which are greater than number of blocks.
 *  - Then the population standard deviation is calculated for that row using formula (sumOfSquares + N * powf(mean, 2.0) - 2 * mean * sumOfTheElements)/N;
 *    Where,
 *        N - Size of the Matrix
 *  - The above values are used to calculate standard score of each element in that row.
 *  - The computed values are stored in the output matrix at their inverse position. This operation is done for all the elements
 *
 *  Steps to compile and execute
    -----------------------------
    1) Go to folder "/home/hpatil2/hw4"
    2) run : qlogin -q interactive.q
    3) run : cd /home/hpatil2/hw4
    4) run : nvcc MatrixNormalizationCuda.cu -o MatrixNormalizationCuda
    5) run : ./MatrixNormalizationCuda 15000 4

    In step 5 : [argument1 (15000) is MATRIX_SIZE, this is mandatory to pass. and maximum value it can take is 15000]
                [argument2 (4) is seed value, this is an optional field]
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

/* Program Parameters */
#define MAXN 15000  /* Max value of N */
int N;  /* Matrix size */

/* Matrices */
volatile float A[MAXN][MAXN], B[MAXN][MAXN];

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void matrixNorm();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
    int seed = 0;  /* Random seed */
    char uid[32]; /*User name */

    /* Read command-line arguments */
    srand(time_seed());  /* Randomize */

    if (argc == 3) {
        seed = atoi(argv[2]);
        srand(seed);
        printf("Random seed = %i\n", seed);
    }
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }
    else {
        printf("Usage: %s <matrix_dimension> [random seed]\n",
               argv[0]);
        exit(0);
    }

    /* Print parameters */
    printf("\nMatrix dimension N = %i.\n", N);
}

/* Initialize A and B*/
void initialize_inputs() {
    int row, col;

    printf("\nInitializing...\n");
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row][col] = (float)rand() / 32768.0;
            B[row][col] = 0.0;
        }
    }

}

/* Print input matrices */
void print_inputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (col = 0; col < N; col++) {
            for (row = 0; row < N; row++) {
                printf("%5.2f%s", A[row][col], (row < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

void print_B() {
    int row, col;

    if (N < 10) {
        printf("\nB =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%1.10f%s", B[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
    }
}

/*
 *  This method calculates sum and square of given block
 */
__global__ void calculateBlockSum(const float *input, float *sumResults, float *squareResults, const size_t n)
{
    __shared__ float smSum[512];
    __shared__ float smSquare[512];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    float x = 0;

    if(i < n) {
        x = input[i];
    }
    smSum[tx] = x;
    smSquare[tx] = x*x;
    __syncthreads();

    int j;
    for(j = blockDim.x / 2; j > 0; j >>= 1) {

        if(tx < j) {
            smSum[tx] = smSum[tx]+ smSum[tx + j];
            smSquare[tx] = smSquare[tx] + smSquare[tx + j];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {

        sumResults[blockIdx.x] = smSum[0];
        squareResults[blockIdx.x] = smSquare[0];
    }
}

/*
 *  This method calculates final sum and square
 */
__global__ void calculateFinalSum(float *sumResults, float *squareResults, const size_t size, float *finalSumResult, float *finalSigmaResult)  {

    __shared__ float smSum[512];
    __shared__ float smSquare[512];
    int tx = threadIdx.x;

    if(tx < size) {
        smSum[tx] = sumResults[tx];
        smSquare[tx] = squareResults[tx];
    }
    __syncthreads();

    int i;
    for(i = size/2; i > 0; i >>= 1) {

        if(tx < i) {
            smSum[tx] = smSum[tx] + smSum[tx + i];
            smSquare[tx] = smSquare[tx] + smSquare[tx + i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {

        finalSumResult[0] = smSum[0];
        finalSigmaResult[0] = smSquare[0];
    }
}

float* calculateSum (float *input, size_t n, float *dMatrixA, float *sumValue, float *sigmaValue, size_t blockSize, size_t totalBlocks, int nextNearestPowerOf2, float *finalSumResult, float *finalSigmaResult) {

    float *results = (float *)malloc(sizeof(float) * 2);

    cudaMemcpy(dMatrixA, input, sizeof(float) * N, cudaMemcpyHostToDevice);
    calculateBlockSum<<<totalBlocks, blockSize>>> (dMatrixA, sumValue, sigmaValue, n);
    calculateFinalSum<<<1,totalBlocks>>>(sumValue, sigmaValue, nextNearestPowerOf2, finalSumResult, finalSigmaResult);

    cudaMemcpy(&results[0], &finalSumResult[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&results[1], &finalSigmaResult[0], sizeof(float), cudaMemcpyDeviceToHost);

    return results;
}

void matrixNorm() {

    int row, column;
    float mu, sigma;
    float *sumValue = 0, *sigmaValue = 0, *dMatrixA = 0;
    float *finalSumResult = 0, *finalSigmaResult = 0;


    printf("Parallel Computing.\n");

    size_t blockSize = 16;
    size_t totalBlocks;

    if(N%blockSize == 0){
        totalBlocks = (N/blockSize);
    } else {
        totalBlocks = (N/blockSize) + 1;
    }

    int nextNearestPowerOf2 = pow(2, ceil(log(totalBlocks)/log(2)));

    cudaMalloc((void**)&sumValue, sizeof(float) * (nextNearestPowerOf2));
    cudaMemset(sumValue, 0.0, sizeof(float) * nextNearestPowerOf2);
    cudaMalloc((void**)&sigmaValue, sizeof(float) * (nextNearestPowerOf2));
    cudaMemset(sigmaValue, 0.0, sizeof(float) * nextNearestPowerOf2);
    cudaMalloc((void**)&dMatrixA, sizeof(float) * N);
    cudaMalloc((void**)&finalSumResult, sizeof(float));
    cudaMalloc((void**)&finalSigmaResult, sizeof(float));

    for (column=0; column < N; column++) {

        mu = 0.0;
        float *result;
        result = calculateSum ((float *)A[column], N, dMatrixA, sumValue, sigmaValue, blockSize, totalBlocks, nextNearestPowerOf2,finalSumResult,finalSigmaResult);
        mu = result[0] / (float) N;
        sigma = (result[1] + N * powf(mu, 2.0) - 2 * mu * result[0])/(float)N;

        for (row=0; row < N; row++) {
            if (sigma == 0.0) {
                B[row][column] = 0.0;
            } else {
                B[row][column] = (A[column][row] - mu) / sigma;
            }
        }
    }
    cudaFree(sumValue);
    cudaFree(sigmaValue);
    cudaFree(dMatrixA);
}


int main(int argc, char **argv) {

    struct timeval etstart, etstop;
    struct timezone tzdummy;
    clock_t etstart2, etstop2;
    unsigned long long usecstart, usecstop;
    struct tms cputstart, cputstop;

    parameters(argc, argv);
    initialize_inputs();
    print_inputs();

    printf("\nStarting clock.\n");
    gettimeofday(&etstart, &tzdummy);
    etstart2 = times(&cputstart);

    matrixNorm();

    gettimeofday(&etstop, &tzdummy);
    etstop2 = times(&cputstop);
    printf("Stopped clock.\n");
    usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
    usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

    print_B();

    printf("\nElapsed time = %g ms.\n",
           (float)(usecstop - usecstart)/(float)1000);

    printf("(CPU times are accurate to the nearest %g ms)\n",
           1.0/(float)CLOCKS_PER_SEC * 1000.0);
    printf("My total CPU time for parent = %g ms.\n",
           (float)( (cputstop.tms_utime + cputstop.tms_stime) -
                    (cputstart.tms_utime + cputstart.tms_stime) ) /
           (float)CLOCKS_PER_SEC * 1000);
    printf("My system CPU time for parent = %g ms.\n",
           (float)(cputstop.tms_stime - cputstart.tms_stime) /
           (float)CLOCKS_PER_SEC * 1000);
    printf("My total CPU time for child processes = %g ms.\n",
           (float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
                    (cputstart.tms_cutime + cputstart.tms_cstime) ) /
           (float)CLOCKS_PER_SEC * 1000);
    printf("--------------------------------------------\n");

    exit(0);
}