//
// Created by Harshavardhan Patil on 10/23/16.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

/*#include <ulocks.h>
 #include <task.h>
 */

char *ID;

/* Program Parameters */
#define MAXN 10000  /* Max value of N */
int N;  /* Matrix size */
int procs;  /* Number of processors to use */
int myid;

/* Matrices and vectors */
float  A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
                * It is this routine that is timed.
                * It is called only on the parent.
                */

/* returns a seed for srand based on the time */
unsigned int time_seed() {
    struct timeval t;
    struct timezone tzdummy;

    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
    int submit = 0;  /* = 1 if submission parameters should be used */
    int seed = 0;  /* Random seed */
    // char uid[L_cuserid + 2]; /*User name */
    char uid[32];
    /* Read command-line arguments */
    //  if (argc != 3) {
    if ( argc == 1 && !strcmp(argv[1], "submit") ) {
        /* Use submission parameters */
        submit = 1;
        N = 4;
        procs = 2;
        //printf("\nSubmission run for \"%s\".\n", cuserid(uid));
        printf("\nSubmission run for \"%s\".\n", uid);
        /*uid = ID;*/
        strcpy(uid,ID);
        srand(randm());
    }
    else {
        if (argc == 2) {
            seed = atoi(argv[1]);
            srand(seed);
            if (myid == 0) printf("\nRandom seed = %i\n", seed);
        }
        else {
            if (myid == 0) printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
                                  argv[0]);
            printf("       %s submit\n", argv[0]);
            exit(0);
        }
    }
    //  }
    /* Interpret command-line args */
    if (!submit) {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
            printf("N = %i is out of range.\n", N);
            exit(0);
        }
    }

    /* Print parameters */
    if (myid == 0) printf("\nMatrix dimension N = %i.\n", N);
    if (myid == 0) printf("Number of processors = %i.\n", procs);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
    int row, col;

    printf("\nInitializing...\n");
    for (col = 0; col < N; col++) {
        for (row = 0; row < N; row++) {
            A[row][col] = (float)rand() / 32768.0;
        }
        B[col] = (float)rand() / 32768.0;
        X[col] = 0.0;
    }
}

/* Print input matrices */
void print_inputs() {
    int row, col;

    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < N; col++) {
            printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
        }
    }
}

void print_X() {
    int row;

    if (N < 10) {
        printf("\nX = [");
        for (row = 0; row < N; row++) {
            printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
        }
    }
}

int main(int argc, char **argv) {
    ID = argv[argc-1];
    argc--;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    printf("\nProcess number %d", myid);
    /* Process program parameters */
    parameters(argc, argv);

    /* Initialize A and B */
    if (myid == 0) {
        initialize_inputs();

        /* Print input matrices */
        print_inputs();
    }
    /* Gaussian Elimination */
    gauss();
    /* Back substitution */
    if (myid == 0) {
        int row, col;
        for (row = N - 1; row >= 0; row--) {
            X[row] = B[row];
            for (col = N-1; col > row; col--) {
                X[row] -= A[row][col] * X[col];
            }
            X[row] /= A[row][row];
        }
        /* Display output */
        print_X();
    }
    MPI_Finalize();
    return 0;
}

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, procs, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss() {
    MPI_Status status;
    MPI_Request request;
    int norm, row, col, i;  /* Normalization row, and zeroing element row and col */
    float multiplier;
    /*Time Variables*/

    double startwtime = 0.0, endwtime;
    MPI_Barrier(MPI_COMM_WORLD);
    if (myid == 0) {
        printf("\nComputing Parallely Using MPI.\n");
        startwtime = MPI_Wtime();
    }
    /* Gaussian elimination */
    for (norm = 0; norm < N - 1; norm++) {
        /* Broadcast A[norm] row and B[norm]*/
        MPI_Bcast(&A[norm][0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        /*Send data from process 0 to other processes*/
        if (myid == 0) {
            for (i = 1; i < procs; i++) {
                /*Send data to corresponding process using static interleaved scheduling*/
                for (row = norm + 1 + i; row < N; row += procs) {
                    MPI_Isend(&A[row], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, &status);
                    MPI_Isend(&B[row], 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request);
                    MPI_Wait(&request, &status);
                }
            }
            /*Gaussian elimination*/
            for (row = norm + 1; row < N; row += procs) {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
            /*Receive the updated data from other processes*/
            for (i = 1; i < procs; i++) {
                for (row = norm + 1 + i; row < N; row += procs) {
                    MPI_Recv(&A[row], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
                    MPI_Recv(&B[row], 1, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &status);
                }
            }
            if (norm == N - 2) {
                endwtime = MPI_Wtime();
                printf("elapsed time = %f\n", endwtime - startwtime);
            }
        }
            /*Receive data from process 0*/
        else {
            for (row = norm + 1 + myid; row < N; row += procs) {
                MPI_Recv(&A[row], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&B[row], 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
                /*Gaussian elimination*/
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
                /*Send back the results*/
                MPI_Isend(&A[row], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);
                MPI_Isend(&B[row], 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request);
                MPI_Wait(&request, &status);
            }
        }
        /*Barrier syncs all processes*/
        MPI_Barrier(MPI_COMM_WORLD);
    }
}