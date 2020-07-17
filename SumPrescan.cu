/*
 * NAME: Tomas Vycas
 *
 * ASSIGNMENT GOALS ACHIEVED:
 ∗ Block scan
 ∗ Full scan for large vectors
 ∗ Bank conflict avoidance optimization (BCAO)
 *
 * TIMINGS (BLOCK_SIZE = 128):
 ∗ Block scan without BCAO = 1.10294 msecs
 ∗ Block scan with BCAO = 0.47206 msecs
 ∗ Full scan without BCAO = 1.39594 msecs
 ∗ Full scan with BCAO = 0.76058 msecs
 *
 * MACHINE:
 * CPU - Intel® Core™ i7-8700 CPU @ 3.20GHz × 12
 * GPU - GeForce RTX 2060
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

// This block size achieved best results
#define BLOCK_SIZE 128

// For BCAO
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define ZERO_BANK_CONFLICTS
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ( ((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)) )
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
// You need extra shared memory space if using BCAO because of
// the padding. Note this is the number of WORDS of padding:
#define EXTRA (CONFLICT_FREE_OFFSET((BLOCK_SIZE * 2 - 1))

// Compares two arrays and outputs if they match or prints the first element that failed the check otherwise
bool compareArrays(int *array1, int *array2, int numElements) {
	for (int i = 0; i < numElements; ++i) {
		if (array1[i] != array2[i]) {
			printf("ARRAY CHECK FAIL at arr1 = %d, arr2 = %d, at index = %d\n", array1[i], array2[i], i);
			return false;
		}
	}
	return true;
}

// Sequential implementation of a full array scan
__host__
void hostFullScan(int *g_idata, int *g_odata, int n) {

	g_odata[0] = 0;
	for (int i = 1; i < n; i++) {
		g_odata[i] = g_odata[i - 1] + g_idata[i - 1];
	}
}

// Outputs prescanned array with BLOCK_SIZE * 2 blocks
// Is unstable on large arrays
__host__
void hostBlockScan(const int *x, int *y , int numElements){
	int num_blocks = 1 + (numElements - 1) / BLOCK_SIZE;

	for (int blk = 0; blk < num_blocks; blk++){
		int blk_start = blk * BLOCK_SIZE*2;
		int blk_end = blk_start + BLOCK_SIZE*2;

		if (blk_end > numElements){
			blk_end = numElements;
		}

		y[blk_start] = 0; // since this is a prescan, not a scan

		for(int j = blk_start + 1; j < blk_end; j++){
			y[j] = x[j-1] + y[j-1];
		}
	}
}

// Takes the output array and for each block i, adds value i from INCR array to every element
__global__
void uniformAdd(int *outputArray, int numElements, int *INCR){
	int index = threadIdx.x + (2 * BLOCK_SIZE) * blockIdx.x;

	int valueToAdd = INCR[blockIdx.x];

	// Each thread sums two elements
	if (index < numElements){
		outputArray[index] += valueToAdd;
	}
	if (index + BLOCK_SIZE < numElements){
		outputArray[index + BLOCK_SIZE] += valueToAdd;
	}
}

// Block prescan that works on any array length on BLOCK_SIZE * 2 length blocks
__global__
void blockPrescan(int *g_idata, int *g_odata, int n, int *SUM)
{
	__shared__ int temp[BLOCK_SIZE << 1]; // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	int blockOffset = BLOCK_SIZE * blockIdx.x * 2;

//	 Copy the correct elements form the global array
	if (blockOffset + (thid * 2) < n){
		temp[thid * 2] = g_idata[blockOffset + (thid * 2)];
	}
	if (blockOffset + (thid * 2) + 1 < n){
		temp[(thid * 2)+1] = g_idata[blockOffset + (thid * 2)+1];
	}

//	 Build sum in place up the tree
	for (int d = BLOCK_SIZE; d > 0; d >>= 1){
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	if (thid == 0) {
		if(SUM != NULL){
			// If doing a FULL scan, save the last value in the SUMS array for later processing
			SUM[blockIdx.x] = temp[(BLOCK_SIZE << 1) - 1];
		}
		temp[(BLOCK_SIZE << 1) - 1] = 0; // clear the last element
	}

//	 Traverse down tree & build scan
	for (int d = 1; d < BLOCK_SIZE << 1; d <<= 1){
		offset >>= 1;
		__syncthreads();

		if (thid < d){
			int ai = offset*((thid * 2)+1)-1;
			int bi = offset*((thid * 2)+2)-1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

//	 Copy the new array back to global array
	__syncthreads();
	if (blockOffset + (thid * 2) < n){
		g_odata[blockOffset + (thid * 2)] = temp[(thid * 2)]; // write results to device memory
	}
	if (blockOffset + (thid * 2) + 1 < n){
		g_odata[blockOffset + ((thid * 2)+1)] = temp[(thid * 2)+1];
	}
}

__host__
void fullPrescan(int *h_x, int *h_y, int numElements) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	size_t size = numElements * sizeof(int);

	// The number of blocks it would take to process the array at each level
	int blocksPerGridL1 = 1 + (numElements - 1) / (BLOCK_SIZE * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (BLOCK_SIZE * 2);
	int blocksPerGridL3 = 1 + blocksPerGridL2 / (BLOCK_SIZE * 2);

	int *d_x = NULL;
	err = cudaMalloc((void **) &d_x, size);
	CUDA_ERROR(err, "Failed to allocate device array x");

	int *d_y = NULL;
	err = cudaMalloc((void**) &d_y, size);
	CUDA_ERROR(err, "Failed to allocate device array y");

	// Only define in here and actually allocate memory to these arrays if needed
	int *d_SUMS_LEVEL1 = NULL;
	int *d_INCR_LEVEL1 = NULL;
	int *d_SUMS_LEVEL2 = NULL;
	int *d_INCR_LEVEL2 = NULL;

	err = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array x from host to device");

	// Create the device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	//-----------------Pick the correct level and execute the kernels----------

	// The correct level is going to be where the SUMS array can be prescanned with only one block
	if(blocksPerGridL1 == 1){
	cudaEventRecord(d_start, 0);

	blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);

	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);
	cudaDeviceSynchronize();

	} else if (blocksPerGridL2 == 1) {

		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to free device array x");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR_LEVEL1");

		// Start timer and the execution of kernels
		cudaEventRecord(d_start, 0);

		blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		// Run a second prescan on the SUMS array
		blockPrescan<<<blocksPerGridL2, BLOCK_SIZE>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, NULL);

		// Add the values of INCR array to the corresponding blocks of the d_y array
		uniformAdd<<<blocksPerGridL1, BLOCK_SIZE>>>(d_y, numElements, d_INCR_LEVEL1);

		// Stop the timer
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();

	} else if (blocksPerGridL3 == 1) {

		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_SUMS_LEVEL2, (BLOCK_SIZE * 2) * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL2");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		err = cudaMalloc((void**) &d_INCR_LEVEL2, (BLOCK_SIZE * 2)* sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		// Start timer and the execution of kernels
		cudaEventRecord(d_start, 0);

		blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		blockPrescan<<<blocksPerGridL2, BLOCK_SIZE>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, d_SUMS_LEVEL2);

		blockPrescan<<<blocksPerGridL3, BLOCK_SIZE>>>(d_SUMS_LEVEL2, d_INCR_LEVEL2, blocksPerGridL2, NULL);

		uniformAdd<<<blocksPerGridL2, BLOCK_SIZE>>>(d_INCR_LEVEL1, blocksPerGridL1, d_INCR_LEVEL2);

		uniformAdd<<<blocksPerGridL1, BLOCK_SIZE>>>(d_y, numElements, d_INCR_LEVEL1);

		// Stop the timer
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}else {
		printf("The array of length = %d is to large for a level 3 FULL prescan\n", numElements);
		exit(EXIT_FAILURE);
	}

	//---------------------------Timing and verification-----------------------

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch fullPrescan");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	int *h_dOutput = (int *)malloc(size);
	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
	if(compareArrays(h_dOutput, h_y, numElements)){
		printf("DEVICE FULL non-BCAO prescan test passed, the scan took %.5f msecs\n", d_msecs);
	}else{
		printf("DEVICE FULL non-BCAO prescan test failed, the scan took %.5f msecs\n", d_msecs);
	}

	//-------------------------------Cleanup-----------------------------------
	// Free device memory
	err = cudaFree(d_x);
	CUDA_ERROR(err, "Failed to free device array x");
	err = cudaFree(d_y);
	CUDA_ERROR(err, "Failed to free device array y");

	// Only need to free these arrays if they were allocated
	if(blocksPerGridL2 == 1 || blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL1");
		err = cudaFree(d_INCR_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL1");
	}
	if(blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL2");
		err = cudaFree(d_INCR_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL2");
	}

	// Destroy device timer events
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

}

// BCAO Block prescan that works on any array length on BLOCK_SIZE * 2 length blocks
__global__
void BCAO_blockPrescan(int *g_idata, int *g_odata, int n, int *SUM)
{
	__shared__ int temp[BLOCK_SIZE * 2 + (BLOCK_SIZE)]; // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	int blockOffset = BLOCK_SIZE * blockIdx.x * 2;

	// Create the correct offsets for BCAO
	int ai = thid;
	int bi = thid + BLOCK_SIZE;

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	// Copy the correct elements form the global array
	if (blockOffset + ai < n){
		temp[ai + bankOffsetA] = g_idata[blockOffset + ai]; // load input into shared memory
	}
	if (blockOffset + bi < n){
		temp[bi + bankOffsetB] = g_idata[blockOffset + bi];
	}

	// Build sum in place up the tree
	for (int d = BLOCK_SIZE; d > 0; d >>= 1){
		__syncthreads();

		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) {
		if(SUM != NULL){
			// If doing a FULL scan, save the last value in the SUMS array for later processing
			SUM[blockIdx.x] = temp[(BLOCK_SIZE * 2) - 1 + CONFLICT_FREE_OFFSET((BLOCK_SIZE * 2) - 1)];
		}
		temp[(BLOCK_SIZE * 2) - 1 + CONFLICT_FREE_OFFSET((BLOCK_SIZE * 2) - 1)] = 0; // clear the last element
	}

	// Traverse down tree & build scan
	for (int d = 1; d < BLOCK_SIZE * 2; d *= 2){
		offset >>= 1;
		__syncthreads();

		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	// Copy the new array back to global array
	__syncthreads();
	if (blockOffset + ai < n){
		g_odata[blockOffset + ai] = temp[ai + bankOffsetA]; // write results to device memory
	}
	if (blockOffset + bi < n){
		g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
	}
}

__host__
void BCAO_fullPrescan(int *h_x, int *h_y, int numElements) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	size_t size = numElements * sizeof(int);

	// The number of blocks it would take to process the array at each level
	int blocksPerGridL1 = 1 + (numElements - 1) / (BLOCK_SIZE * 2);
	int blocksPerGridL2 = 1 + blocksPerGridL1 / (BLOCK_SIZE * 2);
	int blocksPerGridL3 = 1 + blocksPerGridL2 / (BLOCK_SIZE * 2);

	int *d_x = NULL;
	err = cudaMalloc((void **) &d_x, size);
	CUDA_ERROR(err, "Failed to allocate device array x");

	int *d_y = NULL;
	err = cudaMalloc((void**) &d_y, size);
	CUDA_ERROR(err, "Failed to allocate device array y");

	// Only define in here and actually allocate memory to these arrays if needed
	int *d_SUMS_LEVEL1 = NULL;
	int *d_INCR_LEVEL1 = NULL;
	int *d_SUMS_LEVEL2 = NULL;
	int *d_INCR_LEVEL2 = NULL;

	err = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array x from host to device");

	// Create the device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	//-----------------Pick the correct level and execute the kernels----------

	// The correct level is going to be where the SUMS array can be prescanned with only one block
	if(blocksPerGridL1 == 1){
	cudaEventRecord(d_start, 0);

	BCAO_blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);

	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);
	cudaDeviceSynchronize();

	} else if (blocksPerGridL2 == 1) {

		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		// Start timer and the execution of kernels
		cudaEventRecord(d_start, 0);

		BCAO_blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		// Run a second prescan on the SUMS array
		BCAO_blockPrescan<<<blocksPerGridL2, BLOCK_SIZE>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, NULL);

		// Add the values of INCR array to the corresponding blocks of the d_y array
		uniformAdd<<<blocksPerGridL1, BLOCK_SIZE>>>(d_y, numElements, d_INCR_LEVEL1);

		// Stop the timer
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();

	} else if (blocksPerGridL3 == 1) {

		// SUMS and INCR arrays need to be allocated to store intermediate values
		err = cudaMalloc((void**) &d_SUMS_LEVEL1, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL1");

		err = cudaMalloc((void**) &d_SUMS_LEVEL2, blocksPerGridL1 * sizeof(int));
		CUDA_ERROR(err, "Failed to allocate device vector d_SUMS_LEVEL2");

		err = cudaMalloc((void**) &d_INCR_LEVEL1, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		err = cudaMalloc((void**) &d_INCR_LEVEL2, size);
		CUDA_ERROR(err, "Failed to allocate device vector d_INCR");

		// Start timer and the execution of kernels
		cudaEventRecord(d_start, 0);

		BCAO_blockPrescan<<<blocksPerGridL1, BLOCK_SIZE>>>(d_x, d_y, numElements, d_SUMS_LEVEL1);

		BCAO_blockPrescan<<<blocksPerGridL2, BLOCK_SIZE>>>(d_SUMS_LEVEL1, d_INCR_LEVEL1, blocksPerGridL1, d_SUMS_LEVEL2);

		BCAO_blockPrescan<<<blocksPerGridL3, BLOCK_SIZE>>>(d_SUMS_LEVEL2, d_INCR_LEVEL2, blocksPerGridL2, NULL);

		uniformAdd<<<blocksPerGridL2, BLOCK_SIZE>>>(d_INCR_LEVEL1, blocksPerGridL1, d_INCR_LEVEL2);

		uniformAdd<<<blocksPerGridL1, BLOCK_SIZE>>>(d_y, numElements, d_INCR_LEVEL1);

		// Stop the timer
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}else {
		printf("The array of length = %d is to large for a level 3 FULL prescan\n", numElements);
		exit(EXIT_FAILURE);
	}

	//---------------------------Timing and verification-----------------------

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	int *h_dOutput = (int *)malloc(size);
	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
	if(compareArrays(h_dOutput, h_y, numElements)){
		printf("DEVICE FULL BCAO prescan test passed, the scan took %.5f msecs\n", d_msecs);
	}else{
		printf("DEVICE FULL BCAO prescan test failed, the scan took %.5f msecs\n", d_msecs);
	}

	//-------------------------------Cleanup-----------------------------------
	// Free device memory
	err = cudaFree(d_x);
	CUDA_ERROR(err, "Failed to free device array x");
	err = cudaFree(d_y);
	CUDA_ERROR(err, "Failed to free device array y");

	// Only need to free these arrays if they were allocated
	if(blocksPerGridL2 == 1 || blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL1");
		err = cudaFree(d_INCR_LEVEL1);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL1");
	}
	if(blocksPerGridL3 == 1){
		err = cudaFree(d_SUMS_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_SUMS_LEVEL2");
		err = cudaFree(d_INCR_LEVEL2);
		CUDA_ERROR(err, "Failed to free device array d_INCR_LEVEL2");
	}

	// Destroy device timer events
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

}


int main(void) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// For timing
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// Number of elements in the array
	int numElements = 10000000;
	size_t size = numElements * sizeof(int);
    printf("Prescans of arrays of size %d:\n\n", numElements);

	int *h_x = (int *) malloc(size);
	int *h_yBlock = (int *) malloc(size);
	int *h_yFull = (int *) malloc(size);
	int *h_dOutput = (int *) malloc(size);

	if (h_x == NULL || h_yBlock == NULL || h_yFull == NULL || h_dOutput == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	unsigned int seed = 1;

	// Initialize the host array to random integers
	srand(seed);
	for (int i = 0; i < numElements; i++) {
		h_x[i] = rand() % 10;
	}

	//--------------------------Sequential Scans-------------------------------

	sdkStartTimer(&timer);
	hostBlockScan(h_x, h_yBlock, numElements);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("HOST sequential BLOCK scan on took = %.5fmSecs\n", h_msecs);

	sdkStartTimer(&timer);
	hostFullScan(h_x, h_yFull, numElements);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("HOST squential FULL scan took = %.5fmSecs\n\n", h_msecs);

	//--------------------------Redo the input array---------------------------
	// Create a new identical host input array
	// This is needed because with large arrays (and only large arrays)
	// the hostBlockScan() method overrides some of the input array values.
	int *h_xNew = (int *) malloc(size);
	if (h_xNew == NULL) {
		fprintf(stderr, "Failed to allocate host vector!\n");
		exit(EXIT_FAILURE);
	}

	srand(seed);
	for (int i = 0; i < numElements; i++) {
		h_xNew[i] = rand() % 10;
	}

	//--------------------------Device Block Scans------------------------------

	// Create the device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	int *d_x = NULL;
	err = cudaMalloc((void **) &d_x, size);
	CUDA_ERROR(err, "Failed to allocate device array x");

	int *d_y = NULL;
	err = cudaMalloc((void**) &d_y, size);
	CUDA_ERROR(err, "Failed to allocate device array y");

	err = cudaMemcpy(d_x, h_xNew, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array xNew from host to device");

	// Blocks per grid for the block scans
	int blocksPerGrid = 1 + ((numElements - 1) / (BLOCK_SIZE * 2));

	//----------------------Device Non BCAO Block Scan-------------------------

	cudaEventRecord(d_start, 0);
	blockPrescan<<<blocksPerGrid, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch blockPrescan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
//	printf("BLOCK non-BCAO prescan took %.5f msecs\n", d_msecs);

	if(compareArrays(h_dOutput, h_yBlock, numElements)){
		printf("DEVICE BLOCK non-BCAO prescan test passed, the scan took %.5f msecs\n", d_msecs);
	}else{
		printf("DEVICE BLOCK non-BCAO prescan test failed, the scan took %.5f msecs\n", d_msecs);
	}

	//----------------------Device BCAO Block Scan-----------------------------

	cudaEventRecord(d_start, 0);
	BCAO_blockPrescan<<<blocksPerGrid, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch BCAO_blockPrescan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
//	printf("BLOCK BCAO prescan took %.5f msecs\n", d_msecs);

	if(compareArrays(h_dOutput, h_yBlock, numElements)){
		printf("DEVICE BLOCK BCAO prescan test passed, the scan took %.5f msecs\n\n", d_msecs);
	}else{
		printf("DEVICE BLOCK BCAO prescan test failed, the scan took %.5f msecs\n\n", d_msecs);
	}

	// Free device memory as full scan methods will allocate their own memory
	err = cudaFree(d_x);
	CUDA_ERROR(err, "Failed to free device array x");
	err = cudaFree(d_y);
	CUDA_ERROR(err, "Failed to free device array y");

	//--------------------------Device Full Scans------------------------------

	fullPrescan(h_x, h_yFull, numElements);

	BCAO_fullPrescan(h_x, h_yFull, numElements);

	//--------------------------Cleanup----------------------------------------

	// Destroy device timer events
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Delete host timer
	sdkDeleteTimer(&timer);

	// Reset the device
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	// Free host memory
	free(h_x);
	free(h_yBlock);
	free(h_yFull);
	free(h_dOutput);

	printf("\nFinished");

	return 0;
}
