/* nvcc -O2 -o life life.cu
./life <size> <maxgens> */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define DIES  0
#define ALIVE 1
#define OUTPUT_FILENAME "cuda_life_output.txt"

/* allocate row-major two-dimensional array */
int **allocarray(int P, int Q) {
    int *p = NULL;
    int **a = NULL;

    if (P <= 0 || Q <= 0) {
        return NULL;
    }

    if (cudaMallocManaged((void **)&p, (size_t)P * Q * sizeof(int)) != cudaSuccess) {
        return NULL;
    }

    if (cudaMallocManaged((void **)&a, (size_t)P * sizeof(int *)) != cudaSuccess) {
        cudaFree(p);
        return NULL;
    }

    for (int i = 0; i < P; i++) {
        a[i] = &p[i * Q];
    }

    return a;
}

/* free allocated memory */
void freearray(int **a) {
    if (a == NULL) {
        return;
    }

    if (a[0] != NULL) {
        cudaFree(a[0]);
    }

    cudaFree(a);
}

/* print interior N x N cells from an (N+2) x (N+2) array */
void printarray(FILE *out, const int * const *a, int N, int k) {
    if (a == NULL || N <= 0) {
        return;
    }

    fprintf(out, "Life after %d iterations:\n", k);

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            fprintf(out, "%d ", a[i][j]);
        }
        fprintf(out, "\n");
    }

    fprintf(out, "\n");
    fclose(out);
}

/* update each cell based on old values */
__global__ void compute(int **life, int **temp, int N, int *cellsalive, int *changed) {
    int rank_x = blockIdx.x * blockDim.x + threadIdx.x;
    int rank_y = blockIdx.y * blockDim.y + threadIdx.y;

    int j = rank_x + 1;
    int i = rank_y + 1;

    if (i > N || j > N) {
        return;
    }

    int value =
        life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] +
        life[i][j - 1]                       + life[i][j + 1] +
        life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

    int next;
    if (life[i][j] == ALIVE) {
        next = (value == 2 || value == 3) ? ALIVE : DIES;
    } else {
        next = (value == 3) ? ALIVE : DIES;
    }

    temp[i][j] = next;
    atomicAdd(cellsalive, next);

    if (next != life[i][j]) {
        atomicAdd(changed, 1);
    }
}

/*Helper to create output directory if it doesn't exist*/
int check_directory(const char *path) {
    char tmp[PATH_MAX];
    size_t len;

    if (path == NULL || path[0] == '\0') {
        return -1;
    }

    len = strlen(path);
    if (len >= sizeof(tmp)) {
        return -1;
    }

    snprintf(tmp, sizeof(tmp), "%s", path);

    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    for (char *p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
                return -1;
            }
            *p = '/';
        }
    }

    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) {
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
  int N, NTIMES, **life = NULL, **temp = NULL, **ptr = NULL;
  int i, j, k, flag = 1;
  double t1, t2;
  struct timeval tv;
  int *changed = NULL, *cellsalive = NULL;
  FILE *out = NULL;
  char outpath[PATH_MAX];

  if (argc != 4) {
    printf("Usage: %s <size> <max_iterations> <output_directory>\n", argv[0]);
    exit(-1);
  }

  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);
  
  if (check_directory(argv[3]) != 0) {
        perror("Failed to create output directory");
        return -1;
    }

  if (snprintf(outpath, sizeof(outpath), "%s/%s", argv[3], OUTPUT_FILENAME) >= (int)sizeof(outpath)) {
        printf("Output path is too long\n");
        return -1;
    }

    out = fopen(outpath, "w");
    if (out == NULL) {
        perror("Failed to open output file");
        return -1;
    }


  /* Allocate memory for both arrays */
  life = allocarray(N + 2, N + 2);
  temp = allocarray(N + 2, N + 2);
  cudaDeviceSynchronize();

  if (life == NULL || temp == NULL) {
        fprintf(out, "Array allocation failed\n");
        fclose(out);
        freearray(life);
        freearray(temp);
        return -1;
    }

    if (cudaMallocManaged((void **)&cellsalive, sizeof(int)) != cudaSuccess) {
        fprintf(out, "cellsalive allocation failed\n");
        fclose(out);
        freearray(life);
        freearray(temp);
        return -1;
    }

    if (cudaMallocManaged((void **)&changed, sizeof(int)) != cudaSuccess) {
        fprintf(out, "changed allocation failed\n");
        fclose(out);
        cudaFree(cellsalive);
        freearray(life);
        freearray(temp);
        return -1;
    }

  /* Initialize the boundaries of the life matrix */
  for (i = 0; i < N+2; i++) {
    life[0][i] = life[i][0] = life[N+1][i] = life[i][N+1] = DIES ;
    temp[0][i] = temp[i][0] = temp[N+1][i] = temp[i][N+1] = DIES ;
  }

  /* Initialize the life matrix */
  for (i = 1; i < N+1; i++) {
    srand48(54321 | i);
    for (j = 1; j < N+1; j++) {
	      life[i][j] = (drand48() < 0.5) ? ALIVE : DIES;
      }
    }

#ifdef DEBUG1
  /* Display the initialized life matrix */
  printarray(out, life, N, 0);
#endif

dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaDeviceSynchronize();
    gettimeofday(&tv, NULL);
    t1 = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    for (k = 0; k < NTIMES && flag != 0; k++) {
        *changed = 0;
        *cellsalive = 0;

        compute<<<grid, block>>>(life, temp, N, cellsalive, changed);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(out, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            break;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(out, "Kernel execution failed: %s\n", cudaGetErrorString(err));
            break;
        }

        flag = *changed;

    /* copy the new values to the old array */
    ptr = life;
    life = temp;
    temp = ptr;

#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    fprintf(out, "No. of cells whose value changed in iteration %d = %d\n",k+1,flag) ;

    /* Display the life matrix */
    printarray(out, life, N, k + 1);
#endif
  }

  cudaDeviceSynchronize();
  gettimeofday(&tv, NULL);
  t2 = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

#ifdef DEBUG1
  /* Display the life matrix after k iterations */
  printarray(out, life, N, k);
#endif

  fprintf(out, "Time taken %f seconds for %d iterations, cells alive = %d\n",
	t2 - t1, k, *cellsalive);

  cudaFree(cellsalive);
  cudaFree(changed);
  freearray(life);
  freearray(temp);

  fprintf(out, "Program terminates normally\n") ;
  fclose(out);

  return 0;
}