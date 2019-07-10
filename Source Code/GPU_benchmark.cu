// headers
#include <stdio.h>
#include <windows.h>

#include <cuda.h> // for CUDA

#include "helper_timer.h"

// global variables
// odd number 11444777 is deliberate illustration
int iNumberOfArrayElements=11444777;
// vector add variables start
float *hostInput1=NULL;
float *hostInput2=NULL;
float *hostOutput=NULL;
float *gold=NULL;

float *deviceInput1=NULL;
float *deviceInput2=NULL;
float *deviceOutput=NULL;
// vector add variables end

// matrix multiplication variables start
float *hostA=NULL;
float *hostB=NULL;
float *hostC=NULL;
float *CHost=NULL;

float *deviceA=NULL;
float *deviceB=NULL;
float *deviceC=NULL;
// matrix multiplication variables end
#define BLOCK_WIDTH 4

float timeOnCPU;
float timeOnGPU;


void PrintCUDADeviceProperties(void)
{
	// function declarations
	int ConvertSMVersionNumberToCores(int, int);

	// code
	Sleep(1000);
	printf("CUDA INFORMATION :\n");
	printf("===========================================================================\n");
	cudaError_t ret_cuda_rt;
	int dev_count;
	ret_cuda_rt = cudaGetDeviceCount(&dev_count);
	if (ret_cuda_rt != cudaSuccess)
	{
		printf("CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To %s. Exitting Now ...\n", cudaGetErrorString(ret_cuda_rt));
	}
	else if (dev_count == 0)
	{
		printf("There Is No CUDA Supprted Device On This System. Exitting Now ...\n");
		return;
	}
	else
	{
		printf("Total Number Of CUDA Supporting GPU Device/Devices On This System : %d\n", dev_count);
		for (int i = 0; i<dev_count; i++)
		{
			cudaDeviceProp dev_prop;
			int driverVersion = 0, runtimeVersion = 0;

			ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, i);
			if (ret_cuda_rt != cudaSuccess)
			{
				printf("%s in %s at line %d\n", cudaGetErrorString(ret_cuda_rt), __FILE__, __LINE__);
				return;
			}
			printf("\n");
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			
			
			
			
			Sleep(1000);			printf("\n");
			printf("******** CUDA DRIVER AND RUNTIME INFORMATION ********\n");
			printf("=====================================================\n");
			printf("CUDA Driver Version                                  : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
			printf("CUDA Runtime Version                                 : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("=====================================================\n");
			
			
			
			
			Sleep(1000);			printf("\n");
			printf("********** GPU DEVICE GENERAL INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device Number                                    : %d\n", i);
			printf("GPU Device Name                                      : %s\n", dev_prop.name);
			printf("GPU Device Compute Capability                        : %d.%d\n", dev_prop.major, dev_prop.minor);
			printf("GPU Device Clock Rate                                : %d\n", dev_prop.clockRate);
			printf("GPU Device Type                                      : ");
			if (dev_prop.integrated)
				printf("Integrated ( On-Board )\n");
			else
				printf("Discrete ( Card )\n");
			printf("=====================================================\n");
			
			
			
			
			printf("\n");
			Sleep(1000);			printf("********** GPU DEVICE MEMORY INFORMATION ************\n");
			printf("=====================================================\n");
			printf("GPU Device Total Memory                              : %.0f GB = %.0f MB = %llu Bytes\n", ((float)dev_prop.totalGlobalMem / 1048576.0f) / 1024.0f, (float)dev_prop.totalGlobalMem / 1048576.0f, (unsigned long long) dev_prop.totalGlobalMem);
			printf("GPU Device Available Memory                          : %lu Bytes\n", (unsigned long)dev_prop.totalConstMem);
			printf("GPU Device Host Memory Mapping Capability            : ");
			if (dev_prop.canMapHostMemory)
				printf("Yes ( Can Map Host Memory To Device Memory )\n");
			else
				printf("No ( Can Not Map Host Memory To Device Memory )\n");
			printf("=====================================================\n");
			
			
			
			
			printf("\n");
			Sleep(1000);			printf("****** GPU DEVICE MULTIPROCESSOR INFORMATION ********\n");
			printf("=====================================================\n");
			printf("GPU Device Number Of SMProcessors                    : %d\n", dev_prop.multiProcessorCount);
			printf("GPU Device Number Of Cores Per SMProcessors          : %d\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor));
			printf("GPU Device Total Number Of Cores                     : %d\n", ConvertSMVersionNumberToCores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
			printf("GPU Device Shared Memory Per SMProcessor             : %lu\n", (unsigned long)dev_prop.sharedMemPerBlock);
			printf("GPU Device Number Of Registers Per SMProcessor       : %d\n", dev_prop.regsPerBlock);
			printf("=====================================================\n");
			
			
			
			
			printf("\n");
			Sleep(1000);			printf("*********** GPU DEVICE THREAD INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device Maximum Number Of Threads Per SMProcessor : %d\n", dev_prop.maxThreadsPerMultiProcessor);
			printf("GPU Device Maximum Number Of Threads Per Block       : %d\n", dev_prop.maxThreadsPerBlock);
			printf("GPU Device Threads In Warp                           : %d\n", dev_prop.warpSize);
			printf("GPU Device Maximum Thread Dimensions                 : ( %d, %d, %d )\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
			printf("GPU Device Maximum Grid Dimensions                   : ( %d, %d, %d )\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
			printf("=====================================================\n");
			
			
			
			
			
			printf("\n");
			Sleep(1000);			printf("*********** GPU DEVICE DRIVER INFORMATION ***********\n");
			printf("=====================================================\n");
			printf("GPU Device has ECC support                           : %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("GPU Device CUDA Driver Mode ( TCC Or WDDM )          : %s\n", dev_prop.tccDriver ? "TCC ( Tesla Compute Cluster Driver )" : "WDDM ( Windows Display Driver Model )");
			printf("=====================================================\n");
#endif
			printf("***************************************************************************\n");
		}
	}
}

int ConvertSMVersionNumberToCores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        { -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return(nGpuArchCoresPerSM[index - 1].Cores);
}


void cleanup1(void)
{
    // code
    
    // free allocated device-memory
    if(deviceA)
    {
        cudaFree(deviceA);
        deviceA=NULL;
    }
    
    if(deviceB)
    {
        cudaFree(deviceB);
        deviceB=NULL;
    }
    
    if(deviceC)
    {
        cudaFree(deviceC);
        deviceC=NULL;
    }
    
    // free allocated host-memory
    if(hostA)
    {
        free(hostA);
        hostA=NULL;
    }
    
    if(hostB)
    {
        free(hostB);
        hostB=NULL;
    }
    
    if(hostC)
    {
        free(hostC);
        hostC=NULL;
    }
    
    if(CHost)
    {
        free(CHost);
        CHost=NULL;
    }
}

void fillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize)
{
    // code
    int i;
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i)
    {
        pFloatArray[i] = fScale * rand();
    }
}

// *** CUDA KERNEL DEFINITION ***
// global kernel function definition
__global__ void vecAdd(float *in1,float *in2,float *out,int len)
{
    // variable declarations
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    // code
    if(i < len)
    {
        out[i]=in1[i]+in2[i];
    }
}

// global kernel function definition
__global__ void matrixMultiply(float *A,float *B,float *C,int numARows,int numAColumns,int numBRows,int numBColumns,int numCRows,int numCColumns)
{
    // variable declarations
    int row=blockIdx.y * blockDim.y + threadIdx.y;
    int col=blockIdx.x * blockDim.x + threadIdx.x;
    // code
    if((row < numARows) && (col < numBColumns))
    {
        float Cvalue=0.0;
        for(int k=0; k < numAColumns; k++)
        {
            Cvalue +=A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numCColumns + col]=Cvalue;
    }
}
void matMulHost(float *A,float *B,float* C,int iAColumns,int iCRows,int iCColumns)
{
    // code
    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    for(int i=0;i<iCRows;++i)
    {
        for(int j=0;j<iCColumns;++j)
        {
            float sum=0.0f;
            for(int k=0;k<iAColumns;++k)
            {
                float a=A[i * iAColumns + k];
                float b=B[k * iCColumns + j];
                sum += a * b;
            }
            C[i * iCColumns + j] = sum;
        }
    }
    
    // stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
}

//STEP3_Function
void mymatmul(void)
{	
	// variable declarations
    int numARows;
    int numAColumns;
    int numBRows;
    int numBColumns;
    int numCRows;
    int numCColumns;
    int numCHostRows;
    int numCHostColumns;
	
	numARows=128;
    numAColumns=128;
    numBRows=128;
    numBColumns=128;

    numCRows=numARows;
    numCColumns=numBColumns;
    
    numCHostRows=numARows;
    numCHostColumns=numBColumns;

    int sizeA= numARows * numAColumns * sizeof(float);
    int sizeB= numBRows * numBColumns * sizeof(float);
    int sizeC= numCRows * numCColumns * sizeof(float);
    int sizeCHost= numCHostRows * numCHostColumns * sizeof(float);

	// allocate host-memory
    hostA=(float *)malloc(sizeA);
    if(hostA==NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix A.\nExitting ...\n");
        exit(EXIT_FAILURE);
    }
    
    hostB=(float *)malloc(sizeB);
    if(hostB==NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Matrix B.\nExitting ...\n");
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    hostC=(float *)malloc(sizeC);
    if(hostC== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    CHost=(float *)malloc(sizeCHost);
    if(hostC== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Matrix C.\nExitting ...\n");
        cleanup1();
        exit(EXIT_FAILURE);
    }

    // fill above input host vectors with arbitary but hard-coded data
    fillFloatArrayWithRandomNumbers(hostA,numARows * numAColumns);
    fillFloatArrayWithRandomNumbers(hostB,numBRows * numBColumns);
    
    // allocate device-memory
    cudaError_t err=cudaSuccess;
    err=cudaMalloc((void **)&deviceA,sizeA);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceB,sizeB);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceC,sizeC);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    // copy host memory contents to device memory
    err=cudaMemcpy(deviceA,hostA,sizeA,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMemcpy(deviceB,hostB,sizeB,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    // cuda kernel configuration
    dim3 DimGrid=dim3(ceil((int)numCColumns/(int)BLOCK_WIDTH),ceil((int)numCRows/(int)BLOCK_WIDTH),1);
    dim3 DimBlock=dim3(BLOCK_WIDTH,BLOCK_WIDTH,1);

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    matrixMultiply<<<DimGrid,DimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    // copy device memory to host memory
    err=cudaMemcpy(hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup1();
        exit(EXIT_FAILURE);
    }
    
    // results
    matMulHost(hostA,hostB,CHost,numAColumns,numCHostRows,numCHostColumns);
   
    // compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy=true;
    int breakValue=0;
    int i;
    for(i=0;i<numARows * numAColumns;i++)
    {
        float val1 = CHost[i];
        float val2 = hostC[i];
        if(fabs(val1-val2) > epsilon)
        {
            bAccuracy = false;
            breakValue=i;
            break;
        }
    }
    
    if(bAccuracy==false)
    {
        printf("Break Value = %d\n",breakValue);
    }
    
    char str[125];
    if(bAccuracy==true)
        sprintf(str,"%s","Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    else
        sprintf(str,"%s","Not All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    
    printf("1st Matrix Is From 0th Element %.6f To %dth Element %.6f\n",hostA[0], (numARows * numAColumns)-1, hostA[(numARows * numAColumns)-1]);
    printf("2nd Matrix Is From 0th Element %.6f To %dth Element %.6f\n",hostB[0], (numBRows * numBColumns)-1, hostB[(numBRows * numBColumns)-1]);
    printf("Grid Dimension = (%d,1,1) And Block Dimension = (%d,1,1)\n",DimGrid.x,DimBlock.x);
    printf("Multiplication Of Above 2 Matrices Creates 3rd Matrix As :\n");
    printf("3nd Matrix Is From 0th Element %.6f To %dth Element %.6f\n",hostC[0], (numCRows * numCColumns)-1, hostC[(numCRows * numCColumns)-1]);
    printf("The Time Taken To Do Above Multiplication On CPU = %.6f (ms)\n",timeOnCPU);
    printf("The Time Taken To Do Above Multiplication On GPU = %.6f (ms)\n",timeOnGPU);
    printf("%s\n",str);

    // total cleanup
    cleanup1();
}


//MAIN PROGRAMM
int main(int argc,char *argv[])
{
	//clrscr();
    // function declarations
	void PrintCUDADeviceProperties(void);   	//functtion for getting device properties
    void fillFloatArrayWithRandomNumbers(float *, int);
	void matMulHost(float *,float *,float *,int,int,int);
    void vecAddHost(const float *, const float *, float *, int);
    void cleanup(void);
     
	// code start
	//STEP1
	printf("\n********************************************************************************************STEP 1 : GETTING GPU DEVICE PROPERTIES USING CUDA API'S **********\n");
	//STEP 1 : Getting device properties
	PrintCUDADeviceProperties();
	printf("\n\n");
	Sleep(1000);
	
	//STEP2
	printf("\n********************************************************************************************STEP 2 : VECTOR ARRAY ADDITION (CPU AS WELL AS GPU) ******* ARRAY SIZE 11444777 ***************************************************\n");
	Sleep(1000);
	//STEP 2 : Vector addition of on CPU as well as GPU
    // allocate host-memory
    hostInput1=(float *)malloc(sizeof(float) * iNumberOfArrayElements);
    if(hostInput1== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 1.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    hostInput2=(float *)malloc(sizeof(float) * iNumberOfArrayElements);
    if(hostInput2== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Input Array 2.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    hostOutput=(float *)malloc(sizeof(float) * iNumberOfArrayElements);
    if(hostOutput== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Host Output Array.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    gold=(float *)malloc(sizeof(float) * iNumberOfArrayElements);
    if(gold== NULL)
    {
        printf("CPU Memory Fatal Error = Can Not Allocate Enough Memory For Gold Output Array.\nExitting ...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // fill above input host vectors with arbitary but hard-coded data
    fillFloatArrayWithRandomNumbers(hostInput1,iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2,iNumberOfArrayElements);
    
    // allocate device-memory
    cudaError_t err=cudaSuccess;
    err=cudaMalloc((void **)&deviceInput1,sizeof(float) * iNumberOfArrayElements);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceInput2,sizeof(float) * iNumberOfArrayElements);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMalloc((void **)&deviceOutput,sizeof(float) * iNumberOfArrayElements);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // copy host memory contents to device memory
    err=cudaMemcpy(deviceInput1,hostInput1,sizeof(float) * iNumberOfArrayElements,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    err=cudaMemcpy(deviceInput2,hostInput2,sizeof(float) * iNumberOfArrayElements,cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // cuda kernel configuration
    dim3 DimGrid=dim3(ceil(iNumberOfArrayElements/256.0),1,1);
    dim3 DimBlock=dim3(256,1,1);
    
    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    vecAdd<<<DimGrid,DimBlock>>>(deviceInput1,deviceInput2,deviceOutput,iNumberOfArrayElements);
    
    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    
    // copy device memory to host memory
    err=cudaMemcpy(hostOutput,deviceOutput,sizeof(float) * iNumberOfArrayElements,cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess)
    {
        printf("GPU Memory Fatal Error = %s In File Name %s At Line No. %d.\nExitting ...\n",cudaGetErrorString(err),__FILE__,__LINE__);
        cleanup();
        exit(EXIT_FAILURE);
    }
    
    // results
    vecAddHost(hostInput1, hostInput2, gold, iNumberOfArrayElements);
    
    // compare results for golden-host
    const float epsilon = 0.000001f;
    bool bAccuracy=true;
    int breakValue=0;
    int i;
    for(i=0;i<iNumberOfArrayElements;i++)
    {
        float val1 = gold[i];
        float val2 = hostOutput[i];
        if(fabs(val1-val2) > epsilon)
        {
            bAccuracy = false;
            breakValue=i;
            break;
        }
    }
    
    if(bAccuracy==false)
    {
        printf("Break Value = %d\n",breakValue);
    }
    
    char str[125];
    if(bAccuracy==true)
        sprintf(str,"%s","Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    else
        sprintf(str,"%s","Not All Comparison Of Output Arrays On CPU And GPU Are Accurate Within The Limit Of 0.000001");
    
    printf("1st Array Is From 0th Element %.6f To %dth Element %.6f\n",hostInput1[0], iNumberOfArrayElements-1, hostInput1[iNumberOfArrayElements-1]);
    printf("2nd Array Is From 0th Element %.6f To %dth Element %.6f\n",hostInput2[0], iNumberOfArrayElements-1, hostInput2[iNumberOfArrayElements-1]);
    printf("Grid Dimension = (%d,1,1) And Block Dimension = (%d,1,1)\n",DimGrid.x,DimBlock.x);
    printf("Sum Of Each Element From Above 2 Arrays Creates 3rd Array As :\n");
    printf("3nd Array Is From 0th Element %.6f To %dth Element %.6f\n",hostOutput[0], iNumberOfArrayElements-1, hostOutput[iNumberOfArrayElements-1]);
    printf("The Time Taken To Do Above Addition On CPU = %.6f (ms)\n",timeOnCPU);
    printf("The Time Taken To Do Above Addition On GPU = %.6f (ms)\n",timeOnGPU);
    printf("%s\n",str);
    
    // total cleanup
    cleanup();
	Sleep(1000);
	printf("\n\n");
	printf("\n********************************************************************************************STEP 3 : MATRIX MULTIPLICATION 2D ARRAY ******* [128 128] ***************************************************\n");
	Sleep(1000);
	mymatmul();
	return(0);
}


//CLEANUP
void cleanup(void)
{
    // code
    
    // free allocated device-memory
    if(deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1=NULL;
    }
    
    if(deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2=NULL;
    }
    
    if(deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput=NULL;
    }
    
    // free allocated host-memory
    if(hostInput1)
    {
        free(hostInput1);
        hostInput1=NULL;
    }
    
    if(hostInput2)
    {
        free(hostInput2);
        hostInput2=NULL;
    }
    
    if(hostOutput)
    {
        free(hostOutput);
        hostOutput=NULL;
    }
    
    if(gold)
    {
        free(gold);
        gold=NULL;
    }
}


// "Golden" Host processing vector addition function for comparison purposes
void vecAddHost(const float* pFloatData1, const float* pFloatData2, float* pFloatResult, int iNumElements)
{
    int i;
    
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
    
    for (i = 0; i < iNumElements; i++)
    {
        pFloatResult[i] = pFloatData1[i] + pFloatData2[i];
    }
    
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
}
