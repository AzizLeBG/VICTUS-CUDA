#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>  // Added for file I/O
#include <sstream>  // Added for string stream

// SHA-256 Constants
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        cleanup(); \
        exit(1); \
    } \
} while(0)

// SHA-256 constants
__constant__ uint32_t dev_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// Global variables for CUDA resources
uint8_t* d_inputs = nullptr;
uint8_t* d_outputs = nullptr;
uint8_t* d_target_hash = nullptr;
int* d_match_percents = nullptr;

// Structure to hold the result
struct HashResult {
    uint8_t input[32];
    uint8_t hash[32];
    int match_percent;
};

// Clean up CUDA resources
void cleanup() {
    if (d_inputs) cudaFree(d_inputs);
    if (d_outputs) cudaFree(d_outputs);
    if (d_target_hash) cudaFree(d_target_hash);
    if (d_match_percents) cudaFree(d_match_percents);
}

// Function to convert hex string to bytes
void hex_to_bytes(const char* hex_str, uint8_t* bytes, size_t length) {
    for (size_t i = 0; i < length; i++) {
        sscanf(hex_str + 2 * i, "%2hhx", &bytes[i]);
    }
}

// Function to convert bytes to hex string
std::string bytes_to_hex(const uint8_t* bytes, size_t length) {
    std::stringstream ss;
    for (size_t i = 0; i < length; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
    }
    return ss.str();
}

// CUDA kernel for SHA-256 hash calculation
__global__ void sha256_cuda_kernel(uint8_t* inputs, uint8_t* outputs, uint8_t* target_hash, int* match_percents, uint32_t num_attempts) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_attempts) return;
    
    // Input for this thread
    uint8_t* input = &inputs[idx * 32];
    uint8_t* output = &outputs[idx * 32];
    
    // Set up message schedule array
    uint32_t w[64];
    uint32_t a, b, c, d, e, f, g, h, i, j, t1, t2;
    uint32_t hash[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Pre-processing: convert input to big-endian words
    for (i = 0; i < 8; i++) {
        w[i] = (input[i*4] << 24) | (input[i*4+1] << 16) | (input[i*4+2] << 8) | input[i*4+3];
    }
    
    // Pad the message
    w[8] = 0x80000000; // Padding: first byte is 0x80, rest are 0
    for (i = 9; i < 15; i++) {
        w[i] = 0;
    }
    w[15] = 256; // Length in bits (32 bytes * 8 bits)
    
    // Extend the message schedule array
    for (i = 16; i < 64; i++) {
        w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16];
    }
    
    // Initialize working variables
    a = hash[0];
    b = hash[1];
    c = hash[2];
    d = hash[3];
    e = hash[4];
    f = hash[5];
    g = hash[6];
    h = hash[7];
    
    // Main loop
    for (i = 0; i < 64; i++) {
        t1 = h + SIGMA1(e) + CH(e, f, g) + dev_k[i] + w[i];
        t2 = SIGMA0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    
    // Add compressed chunk to current hash value
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
    
    // Store the hash result
    for (i = 0; i < 8; i++) {
        output[i*4] = (hash[i] >> 24) & 0xFF;
        output[i*4+1] = (hash[i] >> 16) & 0xFF;
        output[i*4+2] = (hash[i] >> 8) & 0xFF;
        output[i*4+3] = hash[i] & 0xFF;
    }
    
    // Calculate match percentage with target
    int matches = 0;
    for (i = 0; i < 32; i++) {
        if (output[i] == target_hash[i]) {
            matches++;
        }
    }
    match_percents[idx] = (matches * 100) / 32;
}

// Generate inputs for a batch
void generate_inputs(uint8_t* base_input, uint8_t* h_inputs, uint64_t total_attempts, uint32_t attempts_this_batch) {
    for (uint32_t i = 0; i < attempts_this_batch; i++) {
        // Copy base input
        memcpy(&h_inputs[i * 32], base_input, 32);
        
        // Modify the last 8 bytes with counter
        uint64_t counter = total_attempts + i;
        for (int j = 0; j < 8; j++) {
            h_inputs[i * 32 + 24 + j] = (counter >> (j * 8)) & 0xFF;
        }
    }
}

// Run a single search batch
bool run_batch(uint8_t* base_input, uint8_t* target_hash, uint32_t attempts_per_batch, 
               uint64_t& total_attempts, HashResult& best_result) {
    
    // Allocate host memory
    uint8_t* h_inputs = new uint8_t[attempts_per_batch * 32];
    uint8_t* h_outputs = new uint8_t[attempts_per_batch * 32];
    int* h_match_percents = new int[attempts_per_batch];
    
    // Generate inputs for this batch
    generate_inputs(base_input, h_inputs, total_attempts, attempts_per_batch);
    
    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_inputs, h_inputs, attempts_per_batch * 32, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (attempts_per_batch + blockSize - 1) / blockSize;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    sha256_cuda_kernel<<<numBlocks, blockSize>>>(d_inputs, d_outputs, d_target_hash, d_match_percents, attempts_per_batch);
    
    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_outputs, d_outputs, attempts_per_batch * 32, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_match_percents, d_match_percents, attempts_per_batch * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Process results
    bool found_new_best = false;
    for (uint32_t i = 0; i < attempts_per_batch; i++) {
        if (h_match_percents[i] > best_result.match_percent) {
            best_result.match_percent = h_match_percents[i];
            memcpy(best_result.input, &h_inputs[i * 32], 32);
            memcpy(best_result.hash, &h_outputs[i * 32], 32);
            found_new_best = true;
            
            // Print new best match
            std::cout << "\nNew best match: " << best_result.match_percent << "%" << std::endl;
            std::cout << "Input: " << bytes_to_hex(best_result.input, 32) << std::endl;
            std::cout << "Hash:  " << bytes_to_hex(best_result.hash, 32) << std::endl;
            
            // If we found a 100% match, we can exit
            if (best_result.match_percent == 100) {
                std::cout << "\n*** FOUND EXACT MATCH! ***\n" << std::endl;
                delete[] h_inputs;
                delete[] h_outputs;
                delete[] h_match_percents;
                return true;
            }
        }
    }
    
    // Update counter and display progress
    total_attempts += attempts_per_batch;
    double hash_rate = attempts_per_batch / (elapsed.count() > 0 ? elapsed.count() : 0.0001);
    
    std::cout << "\rTotal attempts: " << total_attempts 
              << " | Best match: " << best_result.match_percent << "% "
              << " | Speed: " << std::fixed << std::setprecision(2) << hash_rate / 1000000 << " MH/s"
              << "                    " << std::flush;
    
    // Clean up host memory
    delete[] h_inputs;
    delete[] h_outputs;
    delete[] h_match_percents;
    
    return false;
}

// Initialize CUDA resources
bool initialize_cuda(uint8_t* target_hash, uint32_t attempts_per_batch) {
    // Choose which GPU to run on
    CHECK_CUDA(cudaSetDevice(0));
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_inputs, attempts_per_batch * 32));
    CHECK_CUDA(cudaMalloc(&d_outputs, attempts_per_batch * 32));
    CHECK_CUDA(cudaMalloc(&d_target_hash, 32));
    CHECK_CUDA(cudaMalloc(&d_match_percents, attempts_per_batch * sizeof(int)));
    
    // Copy target hash to device
    CHECK_CUDA(cudaMemcpy(d_target_hash, target_hash, 32, cudaMemcpyHostToDevice));
    
    return true;
}

// Save the current best result to a file
void save_result(const HashResult& result) {
    std::ofstream file("best_match.txt");
    file << "Match percent: " << result.match_percent << "%" << std::endl;
    file << "Input: " << bytes_to_hex(result.input, 32) << std::endl;
    file << "Hash:  " << bytes_to_hex(result.hash, 32) << std::endl;
    file.close();
}

// Load the previous best result from a file
bool load_result(HashResult& result) {
    std::ifstream file("best_match.txt");
    if (!file) return false;
    
    std::string line;
    std::string input_hex, hash_hex;
    
    while (std::getline(file, line)) {
        if (line.find("Match percent:") != std::string::npos) {
            result.match_percent = std::stoi(line.substr(14));
        } else if (line.find("Input:") != std::string::npos) {
            input_hex = line.substr(7);
        } else if (line.find("Hash:") != std::string::npos) {
            hash_hex = line.substr(7);
        }
    }
    
    if (!input_hex.empty() && !hash_hex.empty()) {
        hex_to_bytes(input_hex.c_str(), result.input, 32);
        hex_to_bytes(hash_hex.c_str(), result.hash, 32);
        return true;
    }
    
    return false;
}

int main(int argc, char** argv) {
    // Default values
    const char* target_hash_str = "0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19";
    std::string input_hex = "0000000000000000000000000000000000000000000000000000000000000000";
    uint32_t attempts_per_batch = 20000000;  // 20 million
    uint32_t num_batches = 100000;           // 100,000 batches = 2 trillion attempts
    bool use_random_input = false;
    bool load_previous = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target_hash_str = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_hex = argv[++i];
        } else if (strcmp(argv[i], "--attempts") == 0 && i + 1 < argc) {
            attempts_per_batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--batches") == 0 && i + 1 < argc) {
            num_batches = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--random") == 0) {
            use_random_input = true;
        } else if (strcmp(argv[i], "--load") == 0) {
            load_previous = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "P160 Simple CUDA Hash Solver\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --target HASH    Target hash to match (hex string)\n"
                      << "  --input HEX      Starting input value (hex string)\n"
                      << "  --attempts N     Number of attempts per batch (default: 20000000)\n"
                      << "  --batches N      Number of batches to run (default: 100000)\n"
                      << "  --random         Start with random input\n"
                      << "  --load           Load previous best result\n"
                      << "  --help           Show this help message\n\n"
                      << "Press Ctrl+C to exit\n";
            return 0;
        }
    }
    
    // Convert target hash to bytes
    uint8_t target_hash[32];
    hex_to_bytes(target_hash_str, target_hash, 32);
    
    // Convert input hex to bytes
    uint8_t base_input[32];
    hex_to_bytes(input_hex.c_str(), base_input, 32);
    
    // Generate random input if requested
    if (use_random_input) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 255);
        
        for (int i = 0; i < 32; i++) {
            base_input[i] = distrib(gen);
        }
        
        input_hex = bytes_to_hex(base_input, 32);
        std::cout << "Generated random input: " << input_hex << std::endl;
    }
    
    // Initialize CUDA
    if (!initialize_cuda(target_hash, attempts_per_batch)) {
        std::cerr << "Failed to initialize CUDA resources." << std::endl;
        return 1;
    }
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Best match tracking
    HashResult best_result;
    best_result.match_percent = 0;
    
    // Load previous best result if requested
    if (load_previous && load_result(best_result)) {
        std::cout << "Loaded previous best match: " << best_result.match_percent << "%" << std::endl;
        std::cout << "Input: " << bytes_to_hex(best_result.input, 32) << std::endl;
        std::cout << "Hash:  " << bytes_to_hex(best_result.hash, 32) << std::endl;
        
        // Use the loaded input as the new base if it's better
        if (best_result.match_percent > 25) {
            std::cout << "Using loaded input as new base." << std::endl;
            memcpy(base_input, best_result.input, 32);
        }
    }
    
    // Display configuration
    std::cout << "P160 Simple CUDA Hash Solver\n"
              << "=========================\n"
              << "Target hash: " << target_hash_str << "\n"
              << "Starting input: " << bytes_to_hex(base_input, 32) << "\n"
              << "Attempts per batch: " << attempts_per_batch << "\n"
              << "Number of batches: " << num_batches << "\n"
              << "Using GPU: " << prop.name << "\n\n"
              << "Press Ctrl+C to stop at any time.\n" << std::endl;
    
    // Main search loop
    uint64_t total_attempts = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        for (uint32_t batch = 0; batch < num_batches; batch++) {
            // Every 10 batches, update base input to best result if it's better
            if (batch > 0 && batch % 10 == 0 && best_result.match_percent > 25) {
                std::cout << "\nUpdating base input to best match so far." << std::endl;
                memcpy(base_input, best_result.input, 32);
            }
            
            // Run a batch of attempts
            bool found_match = run_batch(base_input, target_hash, attempts_per_batch, total_attempts, best_result);
            
            // Save best result periodically
            if (batch % 10 == 0 || found_match) {
                save_result(best_result);
            }
            
            // Exit if perfect match found
            if (found_match) break;
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "\nUnknown error occurred." << std::endl;
    }
    
    // Final statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "\n\n=== Final Results ===\n";
    std::cout << "Total attempts: " << total_attempts << std::endl;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Average speed: " << (total_attempts / elapsed.count()) / 1000000 << " MH/s" << std::endl;
    std::cout << "Best match: " << best_result.match_percent << "%" << std::endl;
    std::cout << "Input: " << bytes_to_hex(best_result.input, 32) << std::endl;
    std::cout << "Hash:  " << bytes_to_hex(best_result.hash, 32) << std::endl;
    
    // Cleanup
    cleanup();
    
    return 0;
}
