/*
 * Created by Harshavardhan Patil on 9/30/16.
 *
 *
 *  Gaussian Elimination using OpenMP :
 *  - The outer loop of Gaussian elimination has data dependency on multiplier.
 *  - So we can not parallelise it. Middle loop has no dependency, so chose middle one to parallelise.
 *  - Middle for loop has been parallelised using pragma omp directives
 *  - variables Matrix, Vector, iterator and Matrix Size are shared
 *  - variables row, column, multiplier are prive
 *
 *
 * Steps to compile and execute
    -----------------------------
    1) Go to folder "/home/hpatil2/hw2"
    2) run : gcc -o GaussianEliminationUsingOpenMP GaussianEliminationUsingOpenMP.c -fopenmp
    3) run : qlogin -q interactive.q
    4) run : cd hw2
    5) run : ./GaussianEliminationUsingOpenMP 10 4

    In step 5 : [argument1 (10) is MATRIX_SIZE, this is mandatory to pass. and maximum value it can take is 20000]
                [argument2 (4) is seed value, this is an optional field]
 *
 *
 */
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
 * Matrix size and maximum value it can have
 */
#define MAXIMUM_MATRIX_SIZE 20000
int MATRIX_SIZE;

volatile float COEFFICIENT[MAXIMUM_MATRIX_SIZE][MAXIMUM_MATRIX_SIZE], VECTOR[MAXIMUM_MATRIX_SIZE];
volatile float RESULT[MAXIMUM_MATRIX_SIZE];

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

/* Gaussian elimination using openMp
 * Outer loop has data dependency so pragma omp parallelism has been used for 1st inner loop
 * Later on, back substitution has been used to calculate Result
 */
void gaussianEliminationUsingOpenMp(){

    int iterator, row, column;
    float multiplier;
    for (iterator = 0; iterator < MATRIX_SIZE - 1; iterator++) {
        #pragma omp parallel for schedule(static) shared(COEFFICIENT, VECTOR, iterator, MATRIX_SIZE) private(row, column, multiplier)
        for (row = iterator + 1; row < MATRIX_SIZE; row++) {
//                int threadId = omp_get_thread_num();
//                printf("Open mp Thread id %d : \n", threadId);
            multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
            for (column = iterator; column < MATRIX_SIZE; column++) {
//                    int threadCount = omp_get_num_threads();
//                    printf("Number of threads %d : \n", threadCount);
                COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
            }
            VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;
        }
    }

    /* Back substitution */
    for (row = MATRIX_SIZE - 1; row >= 0; row--) {
        RESULT[row] = VECTOR[row];
        for (column = MATRIX_SIZE - 1; column > row; column--) {
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

    gaussianEliminationUsingOpenMp();

    /* Stop Clock */
    gettimeofday(&endTime, &dummyTimeZone);
    executionStartTime = (unsigned long long)startTime.tv_sec * 1000000 + startTime.tv_usec;
    executionEndTime = (unsigned long long)endTime.tv_sec * 1000000 + endTime.tv_usec;
    printResult();
    printf("\n Elapsed time = %g ms \n", (float)(executionEndTime - executionStartTime)/(float)1000);
    exit(0);
}