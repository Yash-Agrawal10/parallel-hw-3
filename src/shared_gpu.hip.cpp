#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Convenience constants
constexpr double PI = M_PI;
using Clock = std::chrono::high_resolution_clock;

// Source term function
double f(double x, double y) { return -8 * PI * PI * sin(2 * PI * x) * sin(2 * PI * y); }

// HIP Function Wrapper
void safe(hipError_t err) {
    if (err != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// GPU Kernels
__global__ void update_grid(double* __restrict__ u, double* __restrict__ u_new, double* __restrict__ f_values, int N, double h) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > 0 && row < N - 1 && col > 0 && col < N - 1) {
        int idx = row * N + col;
        u_new[idx] = (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - h * h * f_values[idx]) / 4.0;
    }
}

__global__ void compute_residual(double* __restrict__ u, double* __restrict__ f_values, double* __restrict__ block_max_residuals, int N, double h) {
    extern __shared__ double shared_residuals[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (row > 0 && row < N - 1 && col > 0 && col < N - 1) {
        double y_partial = (u[(row - 1) * N + col] - 2 * u[row * N + col] + u[(row + 1) * N + col]) / (h * h);
        double x_partial = (u[row * N + (col - 1)] - 2 * u[row * N + col] + u[row * N + (col + 1)]) / (h * h);
        double gradient = x_partial + y_partial;
        double residual = fabs(gradient - f_values[row * N + col]);
        shared_residuals[local_idx] = residual;
    } else {
        shared_residuals[local_idx] = 0.0;
    }
    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride /= 2) {
        if (local_idx < stride) {
            if (shared_residuals[local_idx + stride] > shared_residuals[local_idx]) {
                shared_residuals[local_idx] = shared_residuals[local_idx + stride];
            }
        }
        __syncthreads();
    }

    if (local_idx == 0) {
        block_max_residuals[blockIdx.y * gridDim.x + blockIdx.x] = shared_residuals[0];
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments for verbosity and grid size
    bool verbose = false;
    int N = 256;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            try {
                N = std::stoi(arg);
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid argument: " << arg << ". Expected an integer for grid size." << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Using grid size N = " << N << std::endl;

    // Begin timing
    auto start_time = Clock::now();

    // Define constants
    const double start = 0.0;
    const double end = 1.0;
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-3;

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Initialize host memory
    std::vector<double> u_host(N * N, 0.0);
    std::vector<double> u_new_host(N * N, 0.0);
    std::vector<double> f_values_host(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = start + j * h;
            double y = start + i * h;
            f_values_host[i * N + j] = f(x, y);
        }
    }
    std::vector<double> block_max_residuals_host(gridSize.x * gridSize.y, 0.0);

    // Initialize device memory
    auto alloc_start = Clock::now();
    double *u_dev, *u_new_dev, *f_values_dev, *block_max_residuals_dev;
    safe(hipMalloc((void**) &u_dev, N * N * sizeof(double)));
    safe(hipMalloc((void**) &u_new_dev, N * N * sizeof(double)));
    safe(hipMalloc((void**) &f_values_dev, N * N * sizeof(double)));
    safe(hipMalloc((void**) &block_max_residuals_dev, gridSize.x * gridSize.y * sizeof(double)));
    auto alloc_end = Clock::now();
    std::chrono::duration<double, std::milli> alloc_time = alloc_end - alloc_start;
    double alloc_time_ms = alloc_time.count();

    auto copy_init_start = Clock::now();
    safe(hipMemcpy(u_dev, u_host.data(), N * N * sizeof(double), hipMemcpyHostToDevice));
    safe(hipMemcpy(u_new_dev, u_new_host.data(), N * N * sizeof(double), hipMemcpyHostToDevice));
    safe(hipMemcpy(f_values_dev, f_values_host.data(), N * N * sizeof(double), hipMemcpyHostToDevice));
    auto copy_init_end = Clock::now();
    std::chrono::duration<double, std::milli> copy_init_time = copy_init_end - copy_init_start;
    double copy_init_time_ms = copy_init_time.count();

    // Jacobi iteration
    hipEvent_t start_residual, stop_residual, start_update, stop_update;
    double total_residual_time_ms = 0.0, total_update_time_ms = 0.0;
    double total_residual_copy_time_ms = 0.0;
    safe(hipEventCreate(&start_residual));
    safe(hipEventCreate(&stop_residual));
    safe(hipEventCreate(&start_update));
    safe(hipEventCreate(&stop_update));

    int iterations = 0;
    while (true) {
        // Compute residual
        safe(hipEventRecord(start_residual));
        compute_residual<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(double)>>>(
            u_dev, f_values_dev, block_max_residuals_dev, N, h);
        safe(hipEventRecord(stop_residual));
        safe(hipEventSynchronize(stop_residual));
        float residual_time_ms = 0.0f;
        safe(hipEventElapsedTime(&residual_time_ms, start_residual, stop_residual));
        total_residual_time_ms += residual_time_ms; 

        auto residual_copy_start = Clock::now();
        safe(hipMemcpy(block_max_residuals_host.data(), block_max_residuals_dev, gridSize.x * gridSize.y * sizeof(double), hipMemcpyDeviceToHost));
        auto residual_copy_end = Clock::now();
        std::chrono::duration<double, std::milli> residual_copy_time = residual_copy_end - residual_copy_start;
        total_residual_copy_time_ms += residual_copy_time.count();

        double max_residual = 0.0;
        for (double val : block_max_residuals_host) {
            if (val > max_residual) {
                max_residual = val;
            }
        }

        // Optional verbose output
        if (verbose && iterations % 1000 == 0) {
            std::cout << "Iteration " << iterations << ", Max Residual: " << max_residual << std::endl;
        }

        // Check for convergence
        if (max_residual < tolerance) {
            safe(hipMemcpy(u_host.data(), u_dev, N * N * sizeof(double), hipMemcpyDeviceToHost));
            break;
        }

        // Update internal grid points
        safe(hipEventRecord(start_update));
        update_grid<<<gridSize, blockSize>>>(u_dev, u_new_dev, f_values_dev, N, h);
        safe(hipEventRecord(stop_update));
        safe(hipEventSynchronize(stop_update));
        float update_time_ms = 0.0f;
        safe(hipEventElapsedTime(&update_time_ms, start_update, stop_update));
        total_update_time_ms += update_time_ms;

        // Swap grids and increment iteration count
        std::swap(u_dev, u_new_dev);
        ++iterations;
    }

    safe(hipEventDestroy(start_residual));
    safe(hipEventDestroy(stop_residual));
    safe(hipEventDestroy(start_update));
    safe(hipEventDestroy(stop_update));

    // Free device memory
    safe(hipFree(u_dev));
    safe(hipFree(u_new_dev));
    safe(hipFree(f_values_dev));
    safe(hipFree(block_max_residuals_dev));

    // End timing
    auto end_time = Clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Compute bandwidth in GB/s
    // Computing residual -- reads entire u grid once, and reads f_values once
    // Updating grid -- reads entire u grid once, writes entire u_new grid once, and reads f_values once
    double residual_bytes = N * N * sizeof(double) + N * N * sizeof(double);
    double update_bytes = N * N * sizeof(double) * 2 + N * N * sizeof(double);
    double bytes = (residual_bytes + update_bytes) * iterations;
    double bandwidth = bytes / (elapsed.count() * 1e9);

    // Output results
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    std::cout << "GPU Memory Allocation Time: " << alloc_time_ms << " ms" << std::endl;
    std::cout << "GPU Initial Memory Copy Time: " << copy_init_time_ms << " ms" << std::endl;
    std::cout << "Total Residual Computation Time: " << total_residual_time_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "Total Residual Copy Time: " << total_residual_copy_time_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "Total Update Time: " << total_update_time_ms / 1000.0 << " seconds" << std::endl;

    return 0;
}