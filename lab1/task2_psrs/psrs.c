// parallel sorting by regular sampling
// idea: sorting between processors first, then intra-processor sorting
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define num_threads 4
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

    int id;
    int base = n / num_threads; // length of segment(assume it is an integer)
    int count[num_threads] = { 0 }; // number of elements in each segment
    int sample[num_threads*num_threads]; 
    int seperators[num_threads-1]; 
    int processor_array[num_threads][num_samples]={0};  //room for final data

    omp_set_num_threads(num_threads);
    
    #pragma omp parallel shared(base,array,n,sample,seperators,processor_array,count) private(id)
    {   
        //均匀划分,局部排序
        id = omp_get_thread_num();
        MergeSort(array,id*base,(id+1)*base-1);
        count[id] = 0;

        //选取样本
        for (int i=0; i<num_threads; i++)
            sample[id*num_threads+i] = array[id*base+(i+1)*(base/num_threads)];
        #pragma omp barrier

        //样本排序
        #pragma omp master
        {
            MergeSort(sample,0,num_threads*num_threads-1);
            //选择主元
            for (int i=0; i<num_threads-1; i++)
                seperators[i] = sample[(i+1)*num_threads];
        // }
        // #pragma omp barrier

        //主元划分,全局交换

        // method 1: critical section
        // for (int i=0; i<base; i++)
        // {
        //     int flag = 0;
        //     for (int j=0; j<num_threads; j++)
        //     {   
                
        //         #pragma omp critical
        //         {

        //         if (j == num_threads-1 && flag == 0)
        //         {
                        
        //                 processor_array[j][count[j]] = array[id*base+i];  
        //                 count[j]++;
                                     
                      
        //         }
        //         else if (array[id*base+i] <= seperators[j] && flag == 0)
        //         {
                    
                          
        //                 processor_array[j][count[j]] = array[id*base+i];  
        //                 count[j]++; 
        //                 flag = 1;   
                                         
                   
        //         }
        //             }
        //     }
        // }

        // method 2: master handle
        for (int i=0; i<num_samples; i++)
        {
            for (int j=0; j<num_threads; j++)
            {   
                if (j == num_threads-1)
                {
                          
                        processor_array[j][count[j]] = array[i];  
                        count[j]++;
                                     
                      
                }
                else if (array[i] <= seperators[j])
                {
                    
                          
                        processor_array[j][count[j]] = array[i];  
                        count[j]++;    
                                         
                    break;
                }
            }
        }
        }
      
        
        #pragma omp barrier

        //归并排序
        MergeSort(processor_array[id],0,count[id]-1);

        #pragma omp barrier
        
        //writeback
        //count start index
        int index = 0;
        for (int i=0; i<id; i++)
        {
            index += count[i];
        }
        memcpy(array+index,processor_array[id],count[id]*sizeof(int));

        #pragma omp barrier
    }   
}

int main()
{
    int array[num_samples];
    for (int i=0; i<num_samples; i++)
        array[i] = rand() % (10*num_samples);
    double begin,end,time;
    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array[i]);
    putchar('\n');

    int array2[num_samples];
    memcpy(array2,array,sizeof(array));

    // serial
    begin = clock();
    MergeSort(array2,0,num_samples-1);
    end = clock();
    time = (double)(end - begin) / CLOCKS_PER_SEC;
    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array2[i]);
    printf("The serial running time is %lfs\n",time);

    // parallel
    begin = omp_get_wtime();
    PSRS(array, num_samples);
    end = omp_get_wtime();
    time = end - begin;
    // for (int i=0; i<num_samples; i++)
    //     printf("%d ",array[i]);
    printf("The parallel running time is %lfs\n",time);

    // check
    for (int i=0; i<num_samples; i++)
        if (array[i] != array2[i])
        {
            printf("%d ,Wrong answer!\n",i);
            return 0;
        }

    

    return 0;
}

