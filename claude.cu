#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

__global__ void polynomial_kernel(int *buckets_array, uint64_t *primes_array, size_t primes_size)
{
	// XXX: stride doesn't matter with smaller primes_size, since index + stride will exceed size.
	// Consider this 'futureproofing' of sorts?
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	if(index < primes_size) {
		for(int i = index; i < primes_size; i += stride) {
			int c = 0;
			for(uint64_t j = 1; j < primes_array[index]; ++j) {
				if((j * j * j) % primes_array[index] == 2) {
					c++;
				}
			}

			atomicAdd(&buckets_array[c < 2 ? c : 2], 1);
		}
	}
}

#define ARG_PACK(...) __VA_ARGS__
#define RUN_CUDA_FUNC(fn, args) if(fn(args) != cudaSuccess) { printf("%s(%s) failed!\n", #fn, #args); return; }

void polynomial_eval(int *buckets, const std::vector<uint64_t> &primes)
{
	// XXX: Should probably use cudaMallocManaged here, but whatever.

	size_t primes_size = primes.size() * sizeof(uint64_t);
	uint64_t *primes_array = nullptr;
	RUN_CUDA_FUNC(cudaMalloc, ARG_PACK(&primes_array, primes_size));
	RUN_CUDA_FUNC(cudaMemcpy, ARG_PACK(primes_array, primes.data(), primes_size, cudaMemcpyHostToDevice));

	size_t buckets_size = 3 * sizeof(int);
	int *buckets_array = nullptr;
	RUN_CUDA_FUNC(cudaMalloc, ARG_PACK(&buckets_array, buckets_size));
	RUN_CUDA_FUNC(cudaMemset, ARG_PACK(buckets_array, 0, buckets_size));

	// god, cuda sucks.
	int block_size = 256;
	int block_count = (primes.size() + block_size - 1) / block_size;
	polynomial_kernel<<<block_count, block_size>>>(buckets_array, primes_array, primes.size());

	RUN_CUDA_FUNC(cudaDeviceSynchronize,);
	RUN_CUDA_FUNC(cudaMemcpy, ARG_PACK(buckets, buckets_array, buckets_size, cudaMemcpyDeviceToHost));
	
	RUN_CUDA_FUNC(cudaFree, ARG_PACK(primes_array));
	RUN_CUDA_FUNC(cudaFree, ARG_PACK(buckets_array));
}

int main(int argc, char *argv[])
{
	int buckets[3] = {0};

	if (argc != 2) {
		printf("usage: %s primes.txt", argv[0]);
		return 1;
	}

	std::ifstream file(argv[1]);
	if (!file.good()) {
		printf("failed to open %s\n", argv[1]);
		return 1;
	}

	std::string line;
	std::vector<uint64_t> primes;
	while (std::getline(file, line)) {
		primes.push_back(std::stoull(line));
	}
	
	printf("primes size = %zu\n", primes.size());

	printf("starting polynomial_eval()\n");
	auto start = std::chrono::system_clock::now();
	polynomial_eval(buckets, primes);
	auto stop = std::chrono::system_clock::now();
	printf("stopped polynomial_eval()\n");
	
	printf("buckets: %d %d %d\n", buckets[0], buckets[1], buckets[2]);

	auto elapsed = stop - start;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
	auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
	printf("time elapsed: %lld milliseconds (%lld second/s)\n", milliseconds.count(), seconds.count());
}