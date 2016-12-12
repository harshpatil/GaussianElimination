/*
 * Created by Harshavardhan Patil on 10/23/16.
 *
 *  Gaussian Elimination using MPI :
 *  - The outer loop of Gaussian elimination has data dependency on multiplier.
 *  - So we can not parallelise it. Middle loop has no dependency, so choose middle one to parallelise.
 *  - Process 0 will perform both input and output. It sends the input to all other processors
 *     and receives the computed output from them to display final result.
 *  Static Interleaving logic : In this logic each thread starts processing a given iteration, after that it processes iterations
                     which are at an offset of “Number of threads” from it’s current position.

    Steps to compile and execute
    -----------------------------
    1) Go to folder "/home/hpatil2/hw3/q3"
    2) run : mpicc -c GaussianEliminationUsingMPI.c (You will see some warning messages, ignore them)
    3) run : mpicc -o GaussianEliminationUsingMPI GaussianEliminationUsingMPI.c (You will see some warning messages, ignore them)
    4) run : qsub -pe mpich 1 run_gaussian_elimination_mpi.bash (You will get the job Id, make a note of this id.
               Ex : Your job 18000 ("run_gaussian_elimination_mpi.bash") has been submitted)
    5) Once the job is complete output will be generated in run_gaussian_elimination_mpi.bash.o<jobId> file
    6) Program takes 2 arguments, first one is the MATRIX_SIZE and second one is the seed value.
       If you want to change the MATRIX SIZE, open file "run_gaussian_elimination_mpi.bash" using vi editor
       and change the first parameter.
 *
 *
 */
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

/*
 * Matrix size and maximum value it can have
 */
#define MAXIMUM_MATRIX_SIZE 20000
int MATRIX_SIZE;
int processors;  /* Number of processors to use */
int myId;

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
 *  Processor 0 will act as a master, it will broadcast the Coefficient reference row and vector entity for multiplier
 *  Then processor 0 will send (using MPI_Isend()) data to other processors using static interleaving method.
 *  master processor will also calculate value for it's own iterations
 *  Once every slave processor is done with calculation, they will send data back to master processor using MPI_Isend()
 *  Master processor will receive data from all slaves and compute back substitution.
 */

void gaussianEliminationUsingMPI(){

    MPI_Status statusCoefficient[MATRIX_SIZE];
    MPI_Status statusVector[MATRIX_SIZE];
    MPI_Request requestCoefficient[MATRIX_SIZE];
    MPI_Request requestVector[MATRIX_SIZE];
    int iterator, row, column, i;
    float multiplier;


    for (iterator=0; iterator < MATRIX_SIZE-1; iterator++) {

        MPI_Bcast(&COEFFICIENT[iterator][0], MATRIX_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&VECTOR[iterator], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (myId == 0) {
            for (i = 1; i < processors; i++) {
                for (row = iterator + 1 + i; row < MATRIX_SIZE; row = (row+processors)) {
                    MPI_Isend(&COEFFICIENT[row], MATRIX_SIZE, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestCoefficient[row]);
                    MPI_Isend(&VECTOR[row], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestVector[row]);
                }
            }

            for (row = iterator + 1; row < MATRIX_SIZE; row = (row+processors)) {
                multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
                for (column = iterator; column < MATRIX_SIZE; column++) {
                    COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
                }
                VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;
            }

            for (i = 1; i < processors; i++) {
                for (row = iterator + 1 + i; row < MATRIX_SIZE; row = (row+processors)) {
                    MPI_Wait(&requestCoefficient[row], &statusCoefficient[row]);
                    MPI_Recv(&COEFFICIENT[row], MATRIX_SIZE, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &statusCoefficient[row]);
                    MPI_Wait(&requestVector[row], &statusVector[row]);
                    MPI_Recv(&VECTOR[row], 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &statusVector[row]);
                }
            }

        }else {
            for (row = iterator + 1 + myId; row < MATRIX_SIZE; row = (row+processors)) {

                MPI_Recv(&COEFFICIENT[row], MATRIX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &statusCoefficient[row]);
                MPI_Recv(&VECTOR[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &statusVector[row]);

                multiplier = COEFFICIENT[row][iterator] / COEFFICIENT[iterator][iterator];
                for (column = iterator; column < MATRIX_SIZE; column++) {
                    COEFFICIENT[row][column] = COEFFICIENT[row][column] - COEFFICIENT[iterator][column] * multiplier;
                }
                VECTOR[row] = VECTOR[row] - VECTOR[iterator] * multiplier;

                MPI_Send(&COEFFICIENT[row], MATRIX_SIZE, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
                MPI_Send(&VECTOR[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

/*
 * Back substitution
 */
void backSubstitution(){

    int row, column;

    /* Back substitution */
    for (row = MATRIX_SIZE - 1; row >= 0; row--) {
        RESULT[row] = VECTOR[row];
        for (column = MATRIX_SIZE - 1; column > row; column--) {
            RESULT[row] = RESULT[row] - COEFFICIENT[row][column] * RESULT[column];
        }
        RESULT[row] = RESULT[row] / COEFFICIENT[row][row];
    }
}

/*
 * Processor 0 will act as a master processor, it will take input and calculate back substitution
 */
int main(int argc, char **argv) {

    double startTime, endTime;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);
    parameters(argc, argv);

    if (myId == 0) {
        initializeInputValues();
        printf("\nMatrix dimension N = %i.\n", MATRIX_SIZE);
        printInputValues();
    }

    startTime = MPI_Wtime();
    gaussianEliminationUsingMPI();

    if(myId == 0){
        endTime = MPI_Wtime();
        backSubstitution();
        printResult();
        printf("\nTotal time taken by MPI = %f\n", (endTime-startTime));
    }
    MPI_Finalize();
    exit(0);
}

