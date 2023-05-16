#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define num_samples 10000

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
    int group_size,my_rank;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD,&group_size);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

    // print data
    // printf("test 1: %d\n",my_rank);
    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array[i]);
    // putchar('\n');
    
    MPI_Barrier(MPI_COMM_WORLD);
    // getchar();

    // MPI_Bcast(array,num_samples,MPI_INT,0,MPI_COMM_WORLD);

    int base = n / group_size; // length of segment(assume it is an integer)
    int count = 0; // number of elements in each segment
    int *data = (int*)malloc(num_samples*sizeof(int)); // data of this node
    memset(data, 0, num_samples*sizeof(int));
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

    
    // printf("test 2: %d\n",my_rank);
    // for (int i=0; i<count; i++)
    //     printf("%d ",data[i]);
    // putchar('\n');
    

    //选取样本
    for (int i=0; i<group_size; i++)
    {
        sample[i] = data[i*(base/group_size)+1];
    }

    if (my_rank == 0){
        collector = (int*)malloc(group_size*group_size*sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(sample,group_size,MPI_INT,collector,group_size,MPI_INT,0,MPI_COMM_WORLD);

    // printf("test 3: %d\n",my_rank);
    // for (int i=0; i<group_size; i++)
    //     printf("%d ",sample[i]);
    // putchar('\n');

    // if (my_rank == 0){
    //     printf("test 4: %d\n",my_rank);
    //     for (int i=0; i<group_size*group_size; i++)
    //         printf("%d ",collector[i]);
    //     putchar('\n');
    // }
    

    
    if (my_rank == 0)
    {
        //样本排序
        MergeSort(collector,0,group_size*group_size-1);
        //选择主元
        for (int i=0; i<group_size-1; i++)
            seperators[i] = collector[(i+1)*group_size];
        
    }  
    //broadcast seperators 
    MPI_Bcast(seperators,group_size-1,MPI_INT,0,MPI_COMM_WORLD); // broadcast must be executed by all processes

    MPI_Barrier(MPI_COMM_WORLD);

    // printf("test 5: %d\n",my_rank);
    // for (int i=0; i<group_size-1; i++)
    //     printf("%d ",seperators[i]);
    // putchar('\n');

    
    // 划分
    int *buffer = (int*)malloc(group_size*num_samples*sizeof(int));  //room for final data
    int *counter = (int*)malloc(group_size*sizeof(int)); // number of elements in each segment
    memset(counter, 0, group_size*sizeof(int));
    memset(buffer, 0, group_size*num_samples*sizeof(int));
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

    // printf("test 6: %d\n",my_rank);
    // for (int i=0; i<group_size*num_samples; i++)
    //     printf("%d ",buffer[i]);
    // putchar('\n');

    // printf("test 7: %d\n",my_rank);
    // for (int i=0; i<group_size; i++)
    //     printf("%d ",counter[i]);
    // putchar('\n');


    // send counter to other nodes
    int * global_counter = (int*)malloc(group_size*group_size*sizeof(int));
    MPI_Allgather(counter,group_size,MPI_INT,global_counter,group_size,MPI_INT,MPI_COMM_WORLD);

    // printf("test 8: %d\n",my_rank);
    // for (int i=0; i<group_size*group_size; i++)
    //     printf("%d ",global_counter[i]);
    // putchar('\n');

    // send data to other nodes
    for (int i=0; i<group_size; i++)
    {
        MPI_Bsend(&buffer[i*num_samples],counter[i],MPI_INT,i,0,MPI_COMM_WORLD);
        memset(&buffer[i*num_samples], 0, counter[i]*sizeof(int));
        MPI_Recv(&buffer[i*num_samples],global_counter[i*group_size+my_rank],MPI_INT,i,0,MPI_COMM_WORLD,&status);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // printf("test 9: %d\n",my_rank);
    // for (int i=0; i<group_size*num_samples; i++)
    //     printf("%d ",buffer[i]);
    // putchar('\n');

    // move buffer to data
    memset(data, 0, num_samples*sizeof(int));
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
    MPI_Bsend(data,num_samples,MPI_INT,0,0,MPI_COMM_WORLD);

    // printf("test 10: %d\n",my_rank);
    // for (int i=0; i<count; i++)
    //     printf("%d ",data[i]);
    // putchar('\n');
    MPI_Barrier(MPI_COMM_WORLD);

    if (my_rank == 0){
        int * new_array = (int*)malloc(group_size*num_samples*sizeof(int));
        memset(new_array, 0, group_size*num_samples*sizeof(int));
        for (int i=0; i<group_size; i++)
            MPI_Recv(new_array+i*num_samples,num_samples,MPI_INT,i,0,MPI_COMM_WORLD,&status);

        // printf("test 11: %d\n",my_rank);
        // for (int i=0; i<group_size*num_samples; i++)
        //     printf("%d ",new_array[i]);
        // putchar('\n');
        // gather none-zero data into data
        count = 0;
        for (int i=0; i<group_size; i++)
        {
            for (int j=0; j<num_samples; j++)
            {
                if (new_array[i*num_samples + j] != 0)
                {
                    array[count] = new_array[i*num_samples + j];
                    count++;
                }
            }
        }


        // for (int i=0; i<count; i++)
        //     printf("%d ",array[i]);
        // putchar('\n');
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
}

int main(int argc, char **argv)
{
    double begin, end, serial_time, parallel_time;

    // serial
    int *array2;
    array2 = (int*)malloc(num_samples*sizeof(int));
    for (int i=0; i<num_samples; i++)
        array2[i] = rand() % (10*num_samples);    
    begin = clock();
    MergeSort(array2,0,num_samples-1);
    end = clock();
    serial_time = (double)(end - begin) / CLOCKS_PER_SEC;
    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array2[i]);
    printf("The serial running time is %lfs\n",serial_time);


    // parallel
    MPI_Init(&argc,&argv);
    int buf_size = 100000;
    //modoify the buffer size
    // MPI_Comm_set_attr(MPI_COMM_WORLD,MPI_BUFFER_SIZE,&buf_size);
    // generate random array
    int *array;
    array = (int*)malloc(num_samples*sizeof(int));
    memcpy(array,array2,num_samples*sizeof(int));

    // MPI buffer
    int *buffer;
    buffer = (int*)malloc(100000*sizeof(int));
    memset(buffer, 0, 100000*sizeof(int));
    MPI_Buffer_attach(buffer,100000);

    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array[i]);
    // putchar('\n');
    begin = clock();
    PSRS(array, num_samples);
    end = clock();
    MPI_Finalize();
    
    parallel_time = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("The parallel running time is %lfs\n",parallel_time);

    // equality check
    for (int i=0; i<num_samples; i++)
        if (array[i] != array2[i])
        {
            printf("%d ,Wrong answer!\n",i);
            return 0;
        }

    return 0;
}
