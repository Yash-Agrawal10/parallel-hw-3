#include <hip/hip_runtime.h>
#include <mpi.h>

#include <chrono>
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
__global__ void update_grid(
    double* __restrict__ u, double* __restrict__ u_new, double* __restrict__ f_values, 
    int local_height, int N, double h, int global_row_offset) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = global_row_offset + row - 1;
    if (row > 0 && row < local_height - 1 && col > 0 && col < N - 1 && !(global_row == 0) && !(global_row == N -1)) {
        int idx = row * N + col;
        u_new[idx] = (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - h * h * f_values[idx]) / 4.0;
    }
}

__global__ void compute_residual(
    double* __restrict__ u, double* __restrict__ f_values, double* __restrict__ block_max_residuals, 
    int local_height, int N, double h, int global_row_offset) 
{
    extern __shared__ double shared_residuals[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int global_row = global_row_offset + row - 1;
    int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
    if (row > 0 && row < local_height - 1 && col > 0 && col < N - 1 && !(global_row == 0) && !(global_row == N -1)) {
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
    // Set up MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verify MPI is working
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        std::cerr << "Error: MPI is not initialized. Please run this program with mpirun." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    } else {
        if (rank == 0) {
            std::cout << "MPI is initialized with " << size << " processes." << std::endl;
        }
    }

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
                if (rank == 0) {
                    std::cerr << "Invalid argument: " << arg << ". Expected an integer for grid size." << std::endl;
                }
                MPI_Finalize();
                return 1;
            }
        }
    }
    if (rank == 0) {
        std::cout << "Using grid size N = " << N << std::endl;
    }

    // Bind each rank to a GPU
    int num_devices = 0;
    safe(hipGetDeviceCount(&num_devices));
    if (num_devices == 0) {
        if (rank == 0) {
            std::cerr << "No HIP devices found." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        safe(hipSetDevice(rank % num_devices));
        if (rank == 0) {
            std::cout << "Found " << num_devices << " HIP devices." << std::endl;
        }
    }

    // Begin timing
    auto start_time = Clock::now();

    // Define global constants
    const double start = 0.0;
    const double end = 1.0;
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-3;

    // Define local variables
    const int N_local = (rank < N % size) ? (N / size + 1) : (N / size);
    int N_previous = 0;
    for (int r = 0; r < rank; ++r) {
        N_previous += (r < N % size) ? (N / size + 1) : (N / size);
    }
    const int local_height = N_local + 2;

    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (local_height + blockSize.y - 1) / blockSize.y);

    // Initialize the local grid and new grid
    std::vector<double> u_host(local_height * N, 0.0);
    std::vector<double> u_new_host(local_height * N, 0.0);
    std::vector<double> f_values_host(local_height * N, 0.0);
    for (int i = 1; i <= N_local; ++i) {
        int global_i = N_previous + (i - 1);
        double y = start + global_i * h;
        for (int j = 0; j < N; ++j) {
            double x = start + j * h;
            f_values_host[i * N + j] = f(x, y);
        }
    }
    std::vector<double> block_max_residuals_host(gridSize.x * gridSize.y, 0.0);

    // Allocate device memory
    double *u_dev, *u_new_dev, *f_values_dev, *block_max_residuals_dev;
    auto alloc_start = Clock::now();
    safe(hipMalloc(&u_dev, local_height * N * sizeof(double)));
    safe(hipMalloc(&u_new_dev, local_height * N * sizeof(double)));
    safe(hipMalloc(&f_values_dev, local_height * N * sizeof(double)));
    safe(hipMalloc(&block_max_residuals_dev, gridSize.x * gridSize.y * sizeof(double)));
    auto alloc_end = Clock::now();
    std::chrono::duration<double, std::milli> alloc_time = alloc_end - alloc_start;
    double alloc_time_ms = alloc_time.count();

    auto copy_init_start = Clock::now();
    safe(hipMemcpy(u_dev, u_host.data(), local_height * N * sizeof(double), hipMemcpyHostToDevice));
    safe(hipMemcpy(u_new_dev, u_new_host.data(), local_height * N * sizeof(double), hipMemcpyHostToDevice));
    safe(hipMemcpy(f_values_dev, f_values_host.data(), local_height * N * sizeof(double), hipMemcpyHostToDevice));
    auto copy_init_end = Clock::now();
    std::chrono::duration<double, std::milli> copy_init_time = copy_init_end - copy_init_start;
    double copy_init_time_ms = copy_init_time.count();

    // Create events
    hipEvent_t start_residual, stop_residual;
    safe(hipEventCreate(&start_residual));
    safe(hipEventCreate(&stop_residual));
    double total_residual_time_ms = 0.0;
    hipEvent_t start_update, stop_update;
    safe(hipEventCreate(&start_update));
    safe(hipEventCreate(&stop_update));
    double total_update_time_ms = 0.0;
    double total_residual_copy_time_ms = 0.0;

    // Jacobi iteration
    int iterations = 0;
    while (true) {
        // Do communication for boundary data
        int prev = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
        int next = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

        double* send_to_prev = (N_local > 0) ? (u_dev + 1 * N) : nullptr;
        double* recv_from_prev = u_dev + 0 * N;
        double* send_to_next = (N_local > 0) ? (u_dev + (local_height - 2) * N) : nullptr;
        double* recv_from_next = u_dev + (local_height - 1) * N;

        MPI_Request reqs[4];
        int k = 0;

        // Recvs first
        MPI_Irecv(recv_from_prev, N, MPI_DOUBLE, prev, 1, MPI_COMM_WORLD, &reqs[k++]);
        MPI_Irecv(recv_from_next, N, MPI_DOUBLE, next, 0, MPI_COMM_WORLD, &reqs[k++]);

        // Sends
        MPI_Isend(send_to_prev, N, MPI_DOUBLE, prev, 0, MPI_COMM_WORLD, &reqs[k++]);
        MPI_Isend(send_to_next, N, MPI_DOUBLE, next, 1, MPI_COMM_WORLD, &reqs[k++]);

        MPI_Waitall(k, reqs, MPI_STATUSES_IGNORE);

        // Compute local residual
        size_t shared_memory = blockSize.x * blockSize.y * sizeof(double);
        safe(hipEventRecord(start_residual));
        compute_residual<<<gridSize, blockSize, shared_memory>>>(
            u_dev, f_values_dev, block_max_residuals_dev, local_height, N, h, N_previous);
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

        // Reduce to find global maximum residual
        double local_max_residual = 0.0;
        for (double val : block_max_residuals_host) {
            if (val > local_max_residual) {
                local_max_residual = val;
            }
        }

        double global_max_residual = 0.0;
        MPI_Allreduce(&local_max_residual, &global_max_residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Optional verbose output
        if (verbose && rank == 0 && iterations % 1000 == 0) {
            std::cout << "Iteration " << iterations << ", Max Residual: " << global_max_residual << std::endl;
        }

        // Check for convergence
        if (global_max_residual < tolerance) {
            safe(hipMemcpy(u_host.data(), u_dev, local_height * N * sizeof(double), hipMemcpyDeviceToHost));
            break;
        }

        // Update internal grid points
        safe(hipEventRecord(start_update));
        update_grid<<<gridSize, blockSize>>>(u_dev, u_new_dev, f_values_dev, local_height, N, h, N_previous);
        safe(hipEventRecord(stop_update));
        safe(hipEventSynchronize(stop_update));
        float update_time_ms = 0.0f;
        safe(hipEventElapsedTime(&update_time_ms, start_update, stop_update));
        total_update_time_ms += update_time_ms;

        // Swap grids and increment iteration count
        std::swap(u_dev, u_new_dev);
        ++iterations;
    }

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
    if (rank == 0) {
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << "GPU Memory Allocation Time: " << alloc_time_ms << " ms" << std::endl;
        std::cout << "GPU Initial Memory Copy Time: " << copy_init_time_ms << " ms" << std::endl;
        std::cout << "Total Residual Computation Time: " << total_residual_time_ms / 1000.0 << " seconds" << std::endl;
        std::cout << "Total Residual Copy Time: " << total_residual_copy_time_ms / 1000.0 << " seconds" << std::endl;
        std::cout << "Total Update Time: " << total_update_time_ms / 1000.0 << " seconds" << std::endl;
    }

    // Finalize MPI environment
    MPI_Finalize();

    return 0;
}