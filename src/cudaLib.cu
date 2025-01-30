#include <random>
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < size)
		y[tid] = scale * x[tid] + y[tid];
}

float rand_float() {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dt(0, std::numeric_limits<float>::max());
	return dt(rng);
}

int runGpuSaxpy(int vectorSize) {
	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	float* x = (float*) malloc(vectorSize * sizeof(*x));
	float* y = (float*) malloc(vectorSize * sizeof(*y));
	float* result = (float*) malloc(vectorSize * sizeof(*result));
	for (int i = 0; i < vectorSize; i++) {
		x[i] = (float) rand();
		y[i] = (float) rand();
	}
	float scale = rand_float();

	float* x_dev;
	float* y_dev;
	gpuAssert(cudaMalloc(&x_dev, vectorSize * sizeof(*x_dev)), __FILE__, __LINE__, true);
	gpuAssert(cudaMalloc(&y_dev, vectorSize * sizeof(*y_dev)), __FILE__, __LINE__, true);
	gpuAssert(
		cudaMemcpy(x_dev, x, vectorSize * sizeof(*x), cudaMemcpyHostToDevice),
		__FILE__,
		__LINE__,
		true
	);
	gpuAssert(
		cudaMemcpy(y_dev, y, vectorSize * sizeof(*y), cudaMemcpyHostToDevice),
		__FILE__,
		__LINE__,
		true
	);

	int threads = 512;
	saxpy_gpu<<<ceil((float) vectorSize / (float) threads), threads>>>(x_dev, y_dev, scale, vectorSize);
	cudaDeviceSynchronize();

	gpuAssert(
		cudaMemcpy(result, y_dev, vectorSize * sizeof(*result), cudaMemcpyDeviceToHost),
		__FILE__,
		__LINE__,
		true
	);
	gpuAssert(cudaFree(x_dev), __FILE__, __LINE__, true);
	gpuAssert(cudaFree(y_dev), __FILE__, __LINE__, true);
	printf("Found %d Errors\n", verifyVector(x, y, result, scale, vectorSize));
	free(x);
	free(y);
	free(result);
	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	curandState_t rng;
	uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(clock64(), tid, 0, &rng);
	if(tid < pSumSize) {
		pSums[tid] = 0;
		for(int i = 0; i < sampleSize; i++) {
			float x = curand_uniform(&rng);
			float y = curand_uniform(&rng);
			pSums[tid] += (hypotf(x, y) < 1) * 1;
		}
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	totals[tid] = 0;
	for(size_t i = reduceSize * tid; i < (tid + 1) * reduceSize; i++) {
		if(i < pSumSize)
			totals[tid] += pSums[i];
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t block_size = 512;
	block_size = std::min(block_size, generateThreadCount);
	uint64_t* pSums_dev;
	gpuAssert(cudaMalloc(&pSums_dev, generateThreadCount * sizeof(*pSums_dev)), __FILE__, __LINE__, true);
	generatePoints<<<ceil((float) generateThreadCount / (float) block_size), block_size>>>(
		pSums_dev,
		generateThreadCount,
		sampleSize
	);
	uint64_t* totals_dev;
	uint64_t* totals;
	gpuAssert(cudaMalloc(&totals_dev, reduceThreadCount * sizeof(*totals_dev)), __FILE__, __LINE__, true);
	totals = (uint64_t*) malloc(reduceThreadCount * sizeof(*totals));
	cudaDeviceSynchronize();

	block_size = 512;
	block_size = std::min(block_size, reduceThreadCount);
	reduceCounts<<<ceil((float) reduceThreadCount / (float) block_size), block_size>>>(
		pSums_dev,
		totals_dev,
		generateThreadCount,
		reduceThreadCount
	);
	cudaDeviceSynchronize();
	gpuAssert(
		cudaMemcpy(totals, totals_dev, reduceThreadCount * sizeof(*totals_dev), cudaMemcpyDeviceToHost),
		__FILE__,
		__LINE__,
		true
	);
	for(int i = 0; i < reduceThreadCount; i++)
		approxPi += totals[i];
	approxPi = approxPi / (generateThreadCount * sampleSize) * 4.0f;

	cudaFree(totals_dev);
	cudaFree(pSums_dev);
	free(totals);

	return approxPi;
}
