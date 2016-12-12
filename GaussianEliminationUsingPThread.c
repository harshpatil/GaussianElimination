/*
 * Created by Harshavardhan Patil on 9/30/16.
 *
 *  Gaussian Elimination using Pthreads :
 *  - The outer loop of Gaussian elimination has data dependency on multiplier.
 *  - So we can not parallelise it. Middle loop has no dependency, so chose middle one to parallelise.
 *  - If matrix size is greater than 20, the total number of threads is fixed to 20. Else total number of threads is equal to matrix size.
 *  - “ParamStruct” object is used for sending data to thread about the threadId,the outer loop index and total number of threads.
 *
    Skip row logic : In this logic each thread starts processing a given iteration, after that it processes iterations
                     which are at an offset of “Number of threads” from it’s current position.
    Slice row logic : In this logic each thread processes chunk of iterations from a given index upto index+”Number of threads”.

    Used skip row logic to nullify lower traiangle of Matrix and followed this by back substitution to get the result

    Steps to compile and execute
    -----------------------------
    1) Go to folder "/home/hpatil2/hw2"
    2) run : gcc -o GaussianEliminationUsingPThread GaussianEliminationUsingPThread.c -lpthread -mcmodel=medium
    3) run : qlogin -q interactive.q
    4) run : cd hw2
    5) run : ./GaussianEliminationUsingPThread 100 4

    In step 5 : [argument1 (100) is MATRIX_SIZE, this is mandatory to pass. and maximum value it can take is 20000]
                [argument2 (4) is seed value, this is an optional field]
 *
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/time.h>

/*
 * Matrix size and maximum value it can have
 */
#define MAXIMUM_MATRIX_SIZE 20000
int MATRIX_SIZE;

volatile float COEFFICIENT[MAXIMUM_MATRIX_SIZE][MAXIMUM_MATRIX_SIZE], VECTOR[MAXIMUM_MATRIX_SIZE];
volatile float RESULT[MAXIMUM_MATRIX_SIZE];

/*
 * Structure to pass multiple values to pthread function
 */
struct ParamStruct {
    int threadCount;
    int iterator;
    int threadId;
};

unsigned int time_seed() {
    struct timeval t;
    struct timezone dummyTimeZone;
    gettimeofday(&t, &dummyTimeZone);
    return (unsigned int)(t.tv_usec);
}

/*
 * function to read parameter and validate
 * First parameter is the Size of the Matrix
 */
void parameters(int argc, char **argv) {
    int seed = 0;
    srand(time_seed());
    if (argc == 3) {
        seed = atoi(argv[2]);
        srand(seed);
        printf("Random seed = %i\n", seed);
    }
    if (argc >= 2) {
        MATRIX_SIZE = atoi(argv[1]);
        if (MATRIX_SIZE < 1 || MATRIX_SIZE > MAXIMUM_MATRIX_SIZE) {
            printf("N = %i is out of range.\n", MATRIX_SIZE);
            exit(0);
        }
    }
    else {
        printf("Usage: %s <matrix_dimension> [random seed]\n",
               argv[0]);
        exit(0);
    }
    printf("\nMatrix dimension N = %i.\n", MATRIX_SIZE);
}

/*
 * Function to assign input values of Coefficient and Vector
 */
void initializeInputValues() {
    int row, col;

    printf("Initializing...\n");
    for (col = 0; col < MATRIX_SIZE; col++) {
        for (row = 0; row < MATRIX_SIZE; row++) {
            COEFFICIENT[row][col] = (float)rand() / 32768.0;
        }
        VECTOR[col] = (float)rand() / 32768.0;
        RESULT[col] = 0.0;
    }
}

/*
 * Function to print values assigned to Coefficient and Vector
 */
void printInputValues() {
    int row, col;

    if (MATRIX_SIZE < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < MATRIX_SIZE; row++) {
            for (col = 0; col < MATRIX_SIZE; col++) {
                printf("%5.2f%s", COEFFICIENT[row][col], (col < MATRIX_SIZE-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < MATRIX_SIZE; col++) {
            printf("%5.2f%s", VECTOR[col], (col < MATRIX_SIZE-1) ? "; " : "]\n");
        }
    }
}

/*
 * Function to print results
 */
void printResult() {
    int row;

    if (MATRIX_SIZE < 100) {
        printf("\nX = [");
        for (row = 0; row < MATRIX_SIZE; row++) {
            printf("%5.2f%s", RESULT[row], (row < MATRIX_SIZE-1) ? "; " : "]\n");
        }
    }
}

/*
 * This is the function to nullify lower triangle values in Coefficient
 * This is called when thread count is less than 20
 * Each thread in this method is responsible to nullify one element
 */
void *nullifyLowerTriangle(struct ParamStruct *paramStruct) {

    int iterator = paramStruct->iterator;
    float multiplier;
    int row, column;
    row = paramStruct->threadId + iterator + 1;
    if (row < MATRIX_SIZE) {
        multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
        for (column = iterator; column < MATRIX_SIZE; column++) {
            COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
        }
        VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;
    }
    pthread_exit(0);
}

/*
int min(int x, int y){
    if(x>y)
        return x;
    return y;
}
*/

/*
 * Below function is called when thread count is greater than 20
 * Each thread in this method processes iterations = sliceValue (sliceValue = Matrix Size / totalThreads)
 * in one chunk
 */
/*
void *nullifyLowerTriangleUsingSliceRowLogic(struct ParamStruct *paramStruct){

    int iterator = paramStruct->iterator;
    int threadCount = paramStruct->threadCounts;
    int beginningRow = (paramStruct->threadId)*threadCount+1+iterator;
    int endRow = min(startIndex+threadCount,MATRIX_SIZE-1);
    float multiplier;
    int row, column;
    for (row = beginningRow; row < endRow; row++) {
        multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
        for (column = iterator; column < MATRIX_SIZE; column++) {
            COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
        }
        VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;
    }
    pthread_exit(0);
}*/


/*
 * Below function is called when thread count is greater than 20
 * Each thread in this method processes iterations = skipValue (skipValue = Matrix Size / totalThreads)
 * Each thread starts processing the iteration based on it's thread id after which it processes all iterations
 * at an offset of skip value from there
 */
void *nullifyLowerTriangleUsingSkipRowLogic(struct ParamStruct *paramStruct){

    int iterator = paramStruct->iterator;
    int beginningRow = paramStruct->threadId + 1;
    float multiplier;
    int row, column;
    for (row = beginningRow + iterator; row < MATRIX_SIZE; row = row + paramStruct->threadCount) {
        multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
        for (column = iterator; column < MATRIX_SIZE; column++) {
            COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
        }
        VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;
    }
    pthread_exit(0);
}

/*
 * Gaussian elimination method using pThread
 * if the MatrixSize is less than 20, total number of pthreads created is equal to MatrixSize.
 * if MatrixSize is greater than 20, total number of pthreads created is 20.
 * this method calls nullifyLowerTriangle & nullifyMultipleLowerTriangleRows to process inner loop parallely
 * After which it determines the Result using Back Substitution
 */
void gaussianEliminationUsingPThread() {

    int iterator, row, column;
    int threadCount;
    if(MATRIX_SIZE>20){
        threadCount = 20;
        pthread_t thread[threadCount];
        struct ParamStruct *paramStruct = malloc(threadCount*sizeof(struct ParamStruct));
        for (iterator = 0; iterator < MATRIX_SIZE-1; iterator++) {
            int i;
            for(i = 0 ; i < threadCount; i++){
                paramStruct[i].threadCount = threadCount;
                paramStruct[i].iterator = iterator;
                paramStruct[i].threadId = i;
                pthread_create(&thread[i], NULL, nullifyLowerTriangleUsingSkipRowLogic, &paramStruct[i]);
            }
            for (i = 0; i < threadCount ; i++) {
                pthread_join(thread[i], NULL);
            }
        }
    }
    else {
        threadCount = MATRIX_SIZE;
        pthread_t thread[threadCount];
        struct ParamStruct *paramStruct = malloc(threadCount*sizeof(struct ParamStruct));
        for (iterator = 0; iterator < MATRIX_SIZE-1; iterator++) {
            int i;
            for(i = 0 ; i < threadCount;i++) {
                paramStruct[i].threadId=i;
                paramStruct[i].iterator=iterator;
                pthread_create(&thread[i], NULL, nullifyLowerTriangle, &paramStruct[i]);
            }
            for (i = 0; i < threadCount; i++) {
                pthread_join(thread[i], NULL);
            }
        }
    }
    /* Back substitution */
    for (row = MATRIX_SIZE - 1; row >= 0; row--) {
        RESULT[row] = VECTOR[row];
        for (column = MATRIX_SIZE-1; column > row; column--) {
            RESULT[row] = RESULT[row] - COEFFICIENT[row][column] * RESULT[column];
        }
        RESULT[row] = RESULT[row] / COEFFICIENT[row][row];
    }
}

int main(int argc, char **argv) {

    struct timeval startTime, endTime;
    struct timezone dummyTimeZone;
    unsigned long long executionStartTime, executionEndTime;

    parameters(argc, argv);
    initializeInputValues();
    printInputValues();
    /* Start Clock */
    gettimeofday(&startTime, &dummyTimeZone);

    /* Gaussian Elimination using pthread */
    gaussianEliminationUsingPThread();

    /* Stop Clock */
    gettimeofday(&endTime, &dummyTimeZone);
    executionStartTime = (unsigned long long)startTime.tv_sec * 1000000 + startTime.tv_usec;
    executionEndTime = (unsigned long long)endTime.tv_sec * 1000000 + endTime.tv_usec;
    printResult();
    printf("\n Elapsed time = %g ms \n", (float)(executionEndTime - executionStartTime)/(float)1000);
    exit(0);
}
