#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// Convenience constants
constexpr double PI = std::numbers::pi;
using Clock = std::chrono::high_resolution_clock;

// Source term function
double f(double x, double y) { return -8 * PI * PI * sin(2 * PI * x) * sin(2 * PI * y); }

// GPU Kernels
void __global__ update_grid() {}

void __global__ compute_residual() {}

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

    // Define constants
    const double start = 0.0;
    const double end = 1.0;
    const double h = (end - start) / (N - 1);
    const double tolerance = 1e-3;

    // Initialize the grid and new grid
    std::vector<double> u(N * N, 0.0);
    std::vector<double> u_new(N * N, 0.0);
    std::vector<double> f_values(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = start + j * h;
            double y = start + i * h;
            f_values[i * N + j] = f(x, y);
        }
    }

    // Begin timing
    auto start_time = Clock::now();

    // Jacobi iteration
    int iterations = 0;
    while (true) {
        // Compute residual
        double max_residual = 0.0;

        // Optional verbose output
        if (verbose && iterations % 1000 == 0) {
            std::cout << "Iteration " << iterations << ", Max Residual: " << max_residual << std::endl;
        }

        // Check for convergence
        if (max_residual < tolerance) {
            break;
        } 

        // Update internal grid points

        // Swap grids and increment iteration count
        std::swap(u, u_new);
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
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    return 0;
}