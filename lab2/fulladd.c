#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define num_samples 8


void fulladd(int *array, int n)
{
    // init
    MPI_Init(NULL,NULL);
    int group_size,my_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Bcast(array,num_samples,MPI_INT,0,MPI_COMM_WORLD);

    int times = log2(group_size);
    int data = array[my_rank];
    int result = data;
    
    
    
    
    MPI_Finalize();
}

int main()
{
    // generate random array
    int array[num_samples];
    for (int i=0; i<num_samples; i++)
        array[i] = rand() % (2*num_samples);
    for (int i=0; i<num_samples; i++)
        printf("%d ",array[i]);
    putchar('\n');

    fulladd(array, num_samples);

    return 0;
}
