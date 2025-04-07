#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

// SHA-256 Constants
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32-(n))))
#define SHR(x, n) ((x) >> (n))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SIGMA0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define SIGMA1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define sigma0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define sigma1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

// CUDA error checking macro
#define CHECK_CUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        cleanup(); \
        exit(1); \
    } \
}

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

// Result structure
struct HashResult {
    uint8_t input[32];
    uint8_t hash[32];
    int match_percent;
    
    HashResult() : match_percent(0) {
        memset(input, 0, 32);
        memset(hash, 0, 32);
    }
};

// CUDA device pointers
uint8_t* d_inputs = nullptr;
uint8_t* d_outputs = nullptr;
uint8_t* d_target_hash = nullptr;
int* d_match_percents = nullptr;
int* d_strategies = nullptr;  // Added for tracking strategy per input

// Host memory
uint8_t* h_inputs = nullptr;
uint8_t* h_outputs = nullptr;
int* h_match_percents = nullptr;
int* h_strategies = nullptr;  // Added for tracking strategy per input

// --- STRATEGY 1: GUIDED SEARCH ---
struct NeighborhoodSearch {
    uint8_t center_input[32];  // The center point of our search
    int center_match_percent;  // Its match percentage
    int radius;                // How far to explore (in byte distance)
    int remaining_attempts;    // How many more attempts in this neighborhood
    
    NeighborhoodSearch() : center_match_percent(0), radius(3), remaining_attempts(0) {
        memset(center_input, 0, 32);
    }
};

// --- STRATEGY 2: PATTERN LEARNING ---
struct PatternDatabase {
    struct Pattern {
        uint8_t bytes[4];  // We'll track 4-byte patterns
        int position;      // Where in the input this pattern was found
        int success_count; // How many times this pattern helped improve matches
        
        Pattern() : position(0), success_count(0) {
            memset(bytes, 0, 4);
        }
    };
    
    static const int MAX_PATTERNS = 1000;
    Pattern patterns[MAX_PATTERNS];
    int num_patterns;
    
    PatternDatabase() : num_patterns(0) {}
};

// --- STRATEGY 3: ADAPTIVE MUTATION ---
struct MutationStrategy {
    enum Type {
        SMALL_DELTA,   // Small changes to bytes
        RANDOM_BYTE,   // Completely random bytes
        BYTE_SWAP,     // Swap bytes within input
        BIT_FLIP       // Flip individual bits
    };
    
    struct Stats {
        int attempts;
        int improvements;
        float success_rate;
        
        Stats() : attempts(0), improvements(0), success_rate(0.25f) {}
    };
    
    Stats strategies[4];  // Stats for each strategy type
    
    MutationStrategy() {
        for (int i = 0; i < 4; i++) {
            strategies[i].attempts = 0;
            strategies[i].improvements = 0;
            strategies[i].success_rate = 0.25f;  // Start with equal rates
        }
    }
    
    Type select_strategy() {
        // Select strategy based on success rates
        float total = 0;
        for (int i = 0; i < 4; i++) {
            total += strategies[i].success_rate;
        }
        
        float r = (float)rand() / RAND_MAX * total;
        float cumulative = 0;
        
        for (int i = 0; i < 4; i++) {
            cumulative += strategies[i].success_rate;
            if (r <= cumulative) {
                return static_cast<Type>(i);
            }
        }
        
        return SMALL_DELTA;  // Default
    }
    
    void update_stats(Type type, bool improved) {
        strategies[type].attempts++;
        if (improved) {
            strategies[type].improvements++;
        }
        
        // Recalculate success rate
        if (strategies[type].attempts >= 100) {
            strategies[type].success_rate = 
                (float)strategies[type].improvements / strategies[type].attempts;
            
            // Ensure minimum success rate to allow exploration
            if (strategies[type].success_rate < 0.05f) {
                strategies[type].success_rate = 0.05f;
            }
            
            // Reset stats periodically to adapt to changing conditions
            if (strategies[type].attempts >= 1000) {
                strategies[type].attempts = 100;
                strategies[type].improvements = 
                    (int)(strategies[type].success_rate * 100);
            }
        }
    }
};

// Global strategy instances
NeighborhoodSearch g_neighborhood;
PatternDatabase g_patterns;
MutationStrategy g_mutation;

// Global best result
HashResult g_best_result;

// Functions for the added strategies
void generate_neighborhood_input(uint8_t* input, const uint8_t* center, int radius) {
    // Start with the center point
    memcpy(input, center, 32);
    
    // Determine how many bytes to modify (between 1 and radius)
    int bytes_to_modify = 1 + rand() % radius;
    
    // Choose random bytes to modify
    for (int i = 0; i < bytes_to_modify; i++) {
        int pos = rand() % 32;
        
        // Small modification to the byte (within +/- radius)
        int delta = (rand() % (2 * radius + 1)) - radius;
        input[pos] = (input[pos] + delta) & 0xFF;
    }
}

void extract_patterns(const uint8_t* input, int match_percent) {
    // Only extract patterns from good matches
    if (match_percent < 15) return;
    
    // Look at all possible 4-byte patterns in the input
    for (int pos = 0; pos <= 32 - 4; pos++) {
        // Check if this pattern already exists
        bool found = false;
        for (int i = 0; i < g_patterns.num_patterns; i++) {
            if (g_patterns.patterns[i].position == pos && 
                memcmp(g_patterns.patterns[i].bytes, &input[pos], 4) == 0) {
                // Pattern exists - increment success count
                g_patterns.patterns[i].success_count++;
                found = true;
                break;
            }
        }
        
        // Add new pattern if not found and we have space
        if (!found && g_patterns.num_patterns < PatternDatabase::MAX_PATTERNS) {
            PatternDatabase::Pattern new_pattern;
            memcpy(new_pattern.bytes, &input[pos], 4);
            new_pattern.position = pos;
            new_pattern.success_count = 1;
            g_patterns.patterns[g_patterns.num_patterns++] = new_pattern;
        }
    }
}

void apply_patterns(uint8_t* input) {
    // If we don't have enough patterns yet, do nothing
    if (g_patterns.num_patterns < 10) return;
    
    // Randomly apply some patterns from our database
    int num_to_apply = 1 + rand() % 3;  // Apply 1-3 patterns
    
    for (int i = 0; i < num_to_apply; i++) {
        // Choose a pattern weighted by success count
        int total_weight = 0;
        for (int j = 0; j < g_patterns.num_patterns; j++) {
            total_weight += g_patterns.patterns[j].success_count;
        }
        
        if (total_weight == 0) break;
        
        int chosen_weight = rand() % (total_weight + 1);
        int running_weight = 0;
        int chosen_idx = 0;
        
        for (int j = 0; j < g_patterns.num_patterns; j++) {
            running_weight += g_patterns.patterns[j].success_count;
            if (running_weight >= chosen_weight) {
                chosen_idx = j;
                break;
            }
        }
        
        // Apply the chosen pattern
        const PatternDatabase::Pattern& pattern = g_patterns.patterns[chosen_idx];
        memcpy(&input[pattern.position], pattern.bytes, 4);
    }
}

void apply_adaptive_mutation(uint8_t* input, MutationStrategy::Type type) {
    switch (type) {
        case MutationStrategy::SMALL_DELTA: {
            // Small changes to 1-3 bytes
            int num_bytes = 1 + rand() % 3;
            for (int i = 0; i < num_bytes; i++) {
                int pos = rand() % 32;
                int delta = (rand() % 11) - 5;  // -5 to +5
                input[pos] = (input[pos] + delta) & 0xFF;
            }
            break;
        }
        
        case MutationStrategy::RANDOM_BYTE: {
            // Replace 1-2 bytes with random values
            int num_bytes = 1 + rand() % 2;
            for (int i = 0; i < num_bytes; i++) {
                int pos = rand() % 32;
                input[pos] = rand() % 256;
            }
            break;
        }
        
        case MutationStrategy::BYTE_SWAP: {
            // Swap two bytes within the input
            int pos1 = rand() % 32;
            int pos2 = rand() % 32;
            uint8_t temp = input[pos1];
            input[pos1] = input[pos2];
            input[pos2] = temp;
            break;
        }
        
        case MutationStrategy::BIT_FLIP: {
            // Flip 1-3 bits
            int num_bits = 1 + rand() % 3;
            for (int i = 0; i < num_bits; i++) {
                int pos = rand() % 32;
                int bit = rand() % 8;
                input[pos] ^= (1 << bit);
            }
            break;
        }
    }
}

void save_patterns(const char* filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file) {
        file.write(reinterpret_cast<const char*>(&g_patterns.num_patterns), sizeof(int));
        file.write(reinterpret_cast<const char*>(g_patterns.patterns), 
                  sizeof(PatternDatabase::Pattern) * g_patterns.num_patterns);
        std::cout << "Saved " << g_patterns.num_patterns << " patterns to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save patterns to " << filename << std::endl;
    }
}

void load_patterns(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file) {
        file.read(reinterpret_cast<char*>(&g_patterns.num_patterns), sizeof(int));
        file.read(reinterpret_cast<char*>(g_patterns.patterns), 
                 sizeof(PatternDatabase::Pattern) * g_patterns.num_patterns);
        std::cout << "Loaded " << g_patterns.num_patterns << " patterns from " << filename << std::endl;
    } else {
        std::cout << "No pattern file found at " << filename << ", starting fresh" << std::endl;
    }
}

// Helpers for conversion
void hex_to_bytes(const char* hex_str, uint8_t* bytes, size_t length) {
    for (size_t i = 0; i < length; i++) {
        sscanf(hex_str + 2 * i, "%2hhx", &bytes[i]);
    }
}

void bytes_to_hex(const uint8_t* bytes, char* hex_str, size_t length) {
    for (size_t i = 0; i < length; i++) {
        sprintf(hex_str + 2 * i, "%02x", bytes[i]);
    }
    hex_str[length * 2] = '\0';
}

// CUDA kernel for SHA-256
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
    w[8] = 0x80000000;  // Padding: first byte is 0x80, rest are 0
    for (i = 9; i < 15; i++) {
        w[i] = 0;
    }
    w[15] = 256;  // Length in bits (32 bytes * 8 bits)

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

// Resource cleanup
void cleanup() {
    if (d_inputs) cudaFree(d_inputs);
    if (d_outputs) cudaFree(d_outputs);
    if (d_target_hash) cudaFree(d_target_hash);
    if (d_match_percents) cudaFree(d_match_percents);
    if (d_strategies) cudaFree(d_strategies);
    
    if (h_inputs) free(h_inputs);
    if (h_outputs) free(h_outputs);
    if (h_match_percents) free(h_match_percents);
    if (h_strategies) free(h_strategies);
}

// Save current best result
void save_result(const HashResult& result, const char* filename = "best_match.txt") {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    
    char input_hex[65] = {0};
    char hash_hex[65] = {0};
    bytes_to_hex(result.input, input_hex, 32);
    bytes_to_hex(result.hash, hash_hex, 32);
    
    file << "Match percent: " << result.match_percent << "%" << std::endl;
    file << "Input: " << input_hex << std::endl;
    file << "Hash:  " << hash_hex << std::endl;
    
    file.close();
}

// Load best result
bool load_result(HashResult& result, const char* filename = "best_match.txt") {
    std::ifstream file(filename);
    if (!file) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Match percent:") != std::string::npos) {
            result.match_percent = std::stoi(line.substr(14));
        } else if (line.find("Input:") != std::string::npos) {
            hex_to_bytes(line.substr(7).c_str(), result.input, 32);
        } else if (line.find("Hash:") != std::string::npos) {
            hex_to_bytes(line.substr(7).c_str(), result.hash, 32);
        }
    }
    
    return true;
}

// Generate advanced input using all strategies
void generate_advanced_input(uint8_t* input, uint64_t counter, int idx, 
                           bool use_guided, bool use_patterns, bool use_adaptive, 
                           bool use_random, const uint8_t* base_input) {
    // Start with base input
    memcpy(input, base_input, 32);
    
    // Strategy selection
    if (use_guided && g_neighborhood.remaining_attempts > 0) {
        // Use guided search around a successful match
        generate_neighborhood_input(input, g_neighborhood.center_input, g_neighborhood.radius);
        g_neighborhood.remaining_attempts--;
        h_strategies[idx] = -1;  // Special marker for guided search
    } else if (use_random) {
        // Generate completely random input
        for (int i = 0; i < 32; i++) {
            input[i] = rand() % 256;
        }
        h_strategies[idx] = -2;  // Special marker for random
    } else if (use_adaptive) {
        // First set up the base with counter
        for (int j = 0; j < 8; j++) {
            input[24 + j] = (counter >> (j * 8)) & 0xFF;
        }
        
        // Then apply adaptive mutation
        MutationStrategy::Type strategy_type = g_mutation.select_strategy();
        apply_adaptive_mutation(input, strategy_type);
        
        // Store the strategy used for this input
        h_strategies[idx] = static_cast<int>(strategy_type);
    } else {
        // Default strategy - just use counter
        for (int j = 0; j < 8; j++) {
            input[24 + j] = (counter >> (j * 8)) & 0xFF;
        }
        h_strategies[idx] = -3;  // Special marker for counter only
    }
    
    // Optionally apply patterns on top of the selected strategy
    if (use_patterns && g_patterns.num_patterns > 0 && rand() % 100 < 30) {
        apply_patterns(input);
    }
}

// Main function
int main(int argc, char** argv) {
    // Default target hash from p160 site
    const char* target_hash_str = "02E0A8B039282FAF6FE0FD769CFBC4B6B4CF8758BA68220EAC420E32B91DDFA673";
    
    // Default starting input (all zeros)
    std::string input_hex = "0000000000000000000000000000000000000000000000000000000000000000";
    
    // Default parameters
    uint32_t attempts_per_batch = 1000000;
    uint32_t num_batches = 100;
    bool use_random_input = false;
    bool load_state_flag = false;
    
    // New strategy flags
    bool use_guided_search = false;
    bool use_pattern_learning = false;
    bool use_adaptive_mutation = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            target_hash_str = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_hex = argv[++i];
        } else if (strcmp(argv[i], "--attempts") == 0 && i + 1 < argc) {
            attempts_per_batch = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batches") == 0 && i + 1 < argc) {
            num_batches = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--random") == 0) {
            use_random_input = true;
        } else if (strcmp(argv[i], "--load") == 0) {
            load_state_flag = true;
        } else if (strcmp(argv[i], "--guided-search") == 0) {
            use_guided_search = true;
        } else if (strcmp(argv[i], "--pattern-learning") == 0) {
            use_pattern_learning = true;
        } else if (strcmp(argv[i], "--adaptive-mutation") == 0) {
            use_adaptive_mutation = true;
        }
    }
    
    // Set random seed
    srand(time(NULL));
    
    // Convert target hash to bytes
    uint8_t target_hash[32];
    hex_to_bytes(target_hash_str, target_hash, 32);
    
    // Convert input to bytes
    uint8_t base_input[32];
    hex_to_bytes(input_hex.c_str(), base_input, 32);
    
    // If loading, try to load best match
    if (load_state_flag) {
        if (load_result(g_best_result)) {
            std::cout << "Loaded previous best match: " << g_best_result.match_percent << "%" << std::endl;
            memcpy(base_input, g_best_result.input, 32);
        }
    }
    
    // Load patterns if using pattern learning
    if (use_pattern_learning) {
        load_patterns("patterns.dat");
    }
    
    // Initialize neighborhood search if using guided search
    if (use_guided_search && g_best_result.match_percent > 0) {
        memcpy(g_neighborhood.center_input, g_best_result.input, 32);
        g_neighborhood.center_match_percent = g_best_result.match_percent;
        g_neighborhood.radius = max(1, 5 - (g_best_result.match_percent / 10));
        g_neighborhood.remaining_attempts = g_best_result.match_percent * 10000;
    }
    
    // Allocate host memory
    h_inputs = (uint8_t*)malloc(attempts_per_batch * 32);
    h_outputs = (uint8_t*)malloc(attempts_per_batch * 32);
    h_match_percents = (int*)malloc(attempts_per_batch * sizeof(int));
    h_strategies = (int*)malloc(attempts_per_batch * sizeof(int));
    
    if (!h_inputs || !h_outputs || !h_match_percents || !h_strategies) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        cleanup();
        return 1;
    }
    
    // Initialize CUDA
    cudaError_t cudaStatus;
    
    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        cleanup();
        return 1;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_inputs, attempts_per_batch * 32));
    CHECK_CUDA(cudaMalloc(&d_outputs, attempts_per_batch * 32));
    CHECK_CUDA(cudaMalloc(&d_target_hash, 32));
    CHECK_CUDA(cudaMalloc(&d_match_percents, attempts_per_batch * sizeof(int)));
    
    // Copy target hash to device
    CHECK_CUDA(cudaMemcpy(d_target_hash, target_hash, 32, cudaMemcpyHostToDevice));
    
    // Batch processing
    int blockSize = 256;
    int numBlocks = (attempts_per_batch + blockSize - 1) / blockSize;
    
    HashResult best_result;
    if (load_state_flag && g_best_result.match_percent > 0) {
        best_result = g_best_result;
    }
    
    uint32_t total_attempts = 0;
    uint64_t total_hashes = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    int previous_best_match = best_result.match_percent;
    
    for (uint32_t batch = 0; batch < num_batches; batch++) {
        // Adjust batch size if we're near the end
        uint32_t attempts_this_batch = std::min(attempts_per_batch, num_batches * attempts_per_batch - total_attempts);
        
        // Generate inputs for this batch using advanced strategies
        for (uint32_t i = 0; i < attempts_this_batch; i++) {
            uint64_t counter = total_attempts + i;
            generate_advanced_input(&h_inputs[i * 32], counter, i,
                                  use_guided_search, use_pattern_learning, use_adaptive_mutation,
                                  use_random_input, base_input);
        }
        
        // Copy inputs to device
        CHECK_CUDA(cudaMemcpy(d_inputs, h_inputs, attempts_this_batch * 32, cudaMemcpyHostToDevice));
        
        // Launch kernel
        sha256_cuda_kernel<<<numBlocks, blockSize>>>(d_inputs, d_outputs, d_target_hash, d_match_percents, attempts_this_batch);
        
        // Check for kernel launch errors
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            cleanup();
            return 1;
        }
        
        // Wait for kernel to finish
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Copy results back to host
        CHECK_CUDA(cudaMemcpy(h_outputs, d_outputs, attempts_this_batch * 32, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_match_percents, d_match_percents, attempts_this_batch * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Process results
        for (uint32_t i = 0; i < attempts_this_batch; i++) {
            if (h_match_percents[i] > best_result.match_percent) {
                best_result.match_percent = h_match_percents[i];
                memcpy(best_result.input, &h_inputs[i * 32], 32);
                memcpy(best_result.hash, &h_outputs[i * 32], 32);
                
                // Save the best result
                save_result(best_result);
                
                // Update g_best_result for future runs
                g_best_result = best_result;
                
                // Update neighborhood center for guided search
                if (use_guided_search) {
                    memcpy(g_neighborhood.center_input, best_result.input, 32);
                    g_neighborhood.center_match_percent = best_result.match_percent;
                    g_neighborhood.radius = max(1, 5 - (best_result.match_percent / 10));
                    g_neighborhood.remaining_attempts = best_result.match_percent * 10000;
                }
                
                // Extract patterns if using pattern learning
                if (use_pattern_learning) {
                    extract_patterns(best_result.input, best_result.match_percent);
                }
                
                // Print best match info
                char input_hex[65] = {0};
                char hash_hex[65] = {0};
                bytes_to_hex(best_result.input, input_hex, 32);
                bytes_to_hex(best_result.hash, hash_hex, 32);
                
                std::cout << "\nNEW BEST MATCH FOUND:" << std::endl;
                std::cout << "Match percent: " << best_result.match_percent << "%" << std::endl;
                std::cout << "Input: " << input_hex << std::endl;
                std::cout << "Hash:  " << hash_hex << std::endl;
                
                // If we found a 100% match, we're done!
                if (best_result.match_percent == 100) {
                    std::cout << "\nFound a 100% match! Puzzle solved!" << std::endl;
                    cleanup();
                    return 0;
                }
            }
            
            // Update adaptive mutation statistics
            if (use_adaptive_mutation && h_strategies[i] >= 0 && h_strategies[i] < 4) {
                bool improved = (h_match_percents[i] > previous_best_match);
                g_mutation.update_stats(static_cast<MutationStrategy::Type>(h_strategies[i]), improved);
            }
        }
        
        // Update total attempts
        total_attempts += attempts_this_batch;
        total_hashes += attempts_this_batch;
        
        // Calculate hash rate
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        double hash_rate = total_hashes / elapsed.count();
        
        // Print status update
        std::cout << "\rTotal attempts: " << total_attempts 
                  << " | Best match: " << best_result.match_percent << "% "
                  << " | Speed: " << std::fixed << std::setprecision(2) << hash_rate / 1000000 << " MH/s";
        
        // Periodically flush output
        if (batch % 10 == 0) {
            std::cout << std::endl;
            
            // If using adaptive mutation, log strategy effectiveness
            if (use_adaptive_mutation) {
                std::cout << "Mutation strategies effectiveness:" << std::endl;
                for (int i = 0; i < 4; i++) {
                    std::cout << "  Strategy " << i << ": " 
                              << g_mutation.strategies[i].success_rate * 100 << "% success rate" 
                              << std::endl;
                }
            }
            
            // If using guided search, log remaining attempts
            if (use_guided_search) {
                std::cout << "Guided search: " << g_neighborhood.remaining_attempts 
                          << " attempts left in current neighborhood (radius: " 
                          << g_neighborhood.radius << ")" << std::endl;
            }
            
            // If using pattern learning, log pattern count
            if (use_pattern_learning) {
                std::cout << "Pattern database: " << g_patterns.num_patterns 
                          << " patterns collected" << std::endl;
            }
        }
        
        // Update previous best for next iteration
        previous_best_match = best_result.match_percent;
    }
    
    // Save patterns if we used pattern learning
    if (use_pattern_learning) {
        save_patterns("patterns.dat");
    }
    
    // Clean up
    cleanup();
    
    return 0;
}
