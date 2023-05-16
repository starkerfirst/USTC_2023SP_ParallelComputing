// find correct processor index is the most difficult part
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define num_samples 16

// Log2
int Log2(int n)
{
    int i=0;
    while(n>1){
        n /= 2;
        i++;
    }
    return i;
}

// Pow
int Pow(int a, int b)
{
    int result = 1;
    for (int i=0; i<b; i++)
        result *= a;
    return result;
}

// btree scheme
void Btree(int *array, int n)
{
    int group_size,my_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    int times = Log2(group_size);
    int data = array[my_rank];
    int result = data;

    // upcast
    for (int i=0; i<times; i++){
        if (my_rank/Pow(2,i) % 2 == 1 && my_rank % Pow(2,i) == 0 ){
            MPI_Send(&result,1,MPI_INT,my_rank-Pow(2,i),i,MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank/Pow(2,i) % 2 == 0 && my_rank % Pow(2,i) == 0){
            MPI_Recv(&data,1,MPI_INT,my_rank+Pow(2,i),i,MPI_COMM_WORLD,&status);
            result += data;
        }
    }

    // printf("test 1: %d %d\n", my_rank, result);

    // downcast
    for (int i=times-1; i>=0; i--){
        if (my_rank/Pow(2,i) % 2 == 0 && my_rank % Pow(2,i) == 0){
            MPI_Send(&result,1,MPI_INT,my_rank+Pow(2,i),times+i,MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (my_rank/Pow(2,i) % 2 == 1 && my_rank % Pow(2,i) == 0){
            MPI_Recv(&result,1,MPI_INT,my_rank-Pow(2,i),times+i,MPI_COMM_WORLD,&status);
        }
    }

    // print
    printf("btree:\n");
    printf("rank %d: %d\n",my_rank,result);
    putchar('\n');
}

// butterfly scheme
void Butterfly(int *array, int n)
{
    int group_size,my_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    int times = Log2(group_size);
    int data = array[my_rank];
    int result = data;

    for (int i=0; i<times; i++)
    {
        // send
        if(my_rank % Pow(2,i+1) >= Pow(2,i))
            MPI_Send(&result,1,MPI_INT,my_rank-Pow(2,i),i,MPI_COMM_WORLD);
        else
            MPI_Send(&result,1,MPI_INT,my_rank+Pow(2,i),i,MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        
        // recv
        if(my_rank % Pow(2,i+1) >= Pow(2,i))
            MPI_Recv(&data,1,MPI_INT,my_rank-Pow(2,i),i,MPI_COMM_WORLD,&status);
        else
            MPI_Recv(&data,1,MPI_INT,my_rank+Pow(2,i),i,MPI_COMM_WORLD,&status);

        // add
        result += data;
    }

    // print
    printf("butterfly:\n");
    printf("rank %d: %d\n",my_rank,result);
    putchar('\n');
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


    MPI_Init(NULL,NULL);
    Btree(array, num_samples);
    MPI_Barrier(MPI_COMM_WORLD);
    Butterfly(array, num_samples);
    MPI_Finalize();

    return 0;
}
