#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define num_samples 10

// merge sort
void MergeSort(int *a, int p, int r)
{
    int q;
    if(p<r){
        q = (p+r)/2;
        MergeSort(a,p,q);
        MergeSort(a,q+1,r);
    
        // merge
        int i,j,k;
        int n1 = q - p + 1;
        int n2 = r - q;
        int *L,*R;
        L = (int*)malloc((n1+1)*sizeof(int));
        R = (int*)malloc((n2+1)*sizeof(int));
        for(i=0; i<n1; i++)
            L[i] = a[p+i];
        L[i] = 100000; // big M
        for(j=0; j<n2; j++)
            R[j] = a[q+j+1];
        R[j] = 100000;
        i=0,j=0;
        for(k=p; k<=r; k++){
            if(L[i]<=R[j]){
                a[k] = L[i];
                i++;
            }
            else{
                a[k] = R[j];
                j++;
            }
        }
    }
} 

void PSRS(int *array, int n)
{
    // init
    MPI_Init(NULL,NULL);
    int group_size,my_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Bcast(array,num_samples,MPI_INT,0,MPI_COMM_WORLD);

    int base = n / group_size; // length of segment(assume it is an integer)
    int count = 0; // number of elements in each segment
    int *data = (int*)malloc(num_samples*sizeof(int)); // data of this node
    int *collector = NULL; // node 0 collects all samples
    int *sample = (int*)malloc(group_size*sizeof(int)); 
    int *seperators = (int*)malloc((group_size-1)*sizeof(int)); 
    

    //均匀划分,局部排序
    for (int i=0; i<base; i++)
    {
        data[count] = array[my_rank*base+i];
        count++;
    }
    MergeSort(data,0,count-1);
    
    //选取样本
    for (int i=0; i<group_size; i++)
    {
        sample[i] = data[i*(base/group_size)];
    }

    if (my_rank == 0){
        collector = (int*)malloc(group_size*group_size*sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(sample,group_size,MPI_INT,collector,group_size,MPI_INT,0,MPI_COMM_WORLD);

    
    if (my_rank == 0)
    {
        //样本排序
        MergeSort(collector,0,group_size*group_size-1);
        //选择主元
        for (int i=0; i<group_size-1; i++)
            seperators[i] = collector[(i+1)*group_size];
        //broadcast seperators
        MPI_Bcast(seperators,group_size-1,MPI_INT,0,MPI_COMM_WORLD);
    }    

    MPI_Barrier(MPI_COMM_WORLD);
    

    // 划分
    int *buffer = (int*)malloc(group_size*num_samples*sizeof(int));  //room for final data
    int *counter = (int*)malloc(group_size*sizeof(int)); // number of elements in each segment
    memset(counter, 0, group_size*sizeof(int));
    for (int i=0, j=0; i<count; i++)
    {
        for (j=0; j<group_size-1; j++)
        {
            if (data[i] < seperators[j])
            {
                buffer[j*num_samples+counter[j]] = data[i];
                counter[j]++;
                break;
            }
        }
        if (j == group_size-1)
        {
            buffer[j*num_samples+counter[j]] = data[i];
            counter[j]++;
        }
    }

    // send counter to other nodes
    int * global_counter = (int*)malloc(group_size*group_size*sizeof(int));
    MPI_Allgather(counter,group_size,MPI_INT,global_counter,group_size,MPI_INT,MPI_COMM_WORLD);

    // send data to other nodes
    for (int i=0; i<group_size; i++)
    {
        MPI_Send(&buffer[i*num_samples],counter[i],MPI_INT,i,0,MPI_COMM_WORLD);
        MPI_Recv(&buffer[i*num_samples],global_counter[i*group_size+my_rank],MPI_INT,i,0,MPI_COMM_WORLD,&status);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // move buffer to data
    count = 0;
    for (int i=0; i<group_size; i++)
    {
        for (int j=0; j<global_counter[i*group_size+my_rank]; j++)
        {
            data[count] = buffer[i*num_samples+j];
            count++;
        }
    }

    //归并排序
    MergeSort(data,0,count-1);

    //writeback to node 0
    MPI_Send(data,num_samples,MPI_INT,0,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0){
        int * new_array = (int*)malloc(group_size*num_samples*sizeof(int));
        for (int i=0; i<group_size; i++)
            MPI_Recv(new_array+i*num_samples,num_samples,MPI_INT,i,0,MPI_COMM_WORLD,&status);
        // gather none-zero data into data
        count = 0;
        for (int i=0; i<group_size; i++)
        {
            for (int j=0; j<num_samples; j++)
            {
                if (new_array[i*num_samples + j] != 0)
                {
                    data[count] = new_array[i*num_samples + j];
                    count++;
                }
            }
        }

        MergeSort(data,0,count-1);

        for (int i=0; i<count; i++)
            printf("%d ",data[i]);
    }
    
    MPI_Finalize();
}

int main()
{
    // generate random array
    int array[num_samples];
    for (int i=0; i<num_samples; i++)
        array[i] = rand() % (10*num_samples);
    for (int i=0; i<num_samples; i++)
        printf("%d ",array[i]);
    putchar('\n');

    PSRS(array, num_samples);

    return 0;
}
