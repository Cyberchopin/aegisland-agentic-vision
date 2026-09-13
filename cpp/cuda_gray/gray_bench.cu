// First-kernel experiment: no tiling, fusion, streams, or GPU pipeline refactor.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

extern "C" __global__ void bgr_to_gray(
    const unsigned char* src, unsigned char* dst, int width, int height,
    size_t src_stride, size_t dst_stride) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const unsigned char* pixel = src + y * src_stride + 3 * x;
    // OpenCV's 15-bit integer BGR8 grayscale convention; validated against
    // the installed cv2 implementation, not assumed equivalent to floating point.
    const unsigned int value = 3735u * pixel[0] + 19235u * pixel[1] + 9798u * pixel[2];
    dst[y * dst_stride + x] = static_cast<unsigned char>((value + 16384u) >> 15);
}

struct Resources {
    unsigned char* src = nullptr;
    unsigned char* dst = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    ~Resources() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (src) cudaFree(src);
        if (dst) cudaFree(dst);
    }
};

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> args;
        if ((argc - 1) % 2) throw std::runtime_error("arguments must be --name value pairs");
        for (int i = 1; i < argc; i += 2) args[argv[i]] = argv[i + 1];
        auto integer = [&](const char* key, int fallback) {
            return args.count(key) ? std::stoi(args.at(key)) : fallback;
        };
        const int width = integer("--width", 0), height = integer("--height", 0);
        const int warmup = integer("--warmup", 10), iterations = integer("--iterations", 100);
        const int bx = integer("--block-x", 32), by = integer("--block-y", 8);
        const int source_stride = integer("--stride", width * 3);
        if (width < 1 || height < 1 || width > 16384 || height > 16384 ||
            source_stride < width * 3 || source_stride > width * 3 + 4096 ||
            warmup < 0 || iterations < 1 || bx < 1 || by < 1 || bx > 1024 || by > 1024 ||
            bx * by > 1024) throw std::runtime_error("invalid shape/count/block/stride");
        const size_t src_stride = source_stride, dst_stride = width + 13;
        std::vector<unsigned char> input(src_stride * height), output(dst_stride * height);
        std::ifstream in(args.at("--input"), std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(input.data()), input.size()) ||
            in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("wrong input size");
        check(cudaSetDevice(0));
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        Resources r;
        check(cudaMalloc(reinterpret_cast<void**>(&r.src), input.size()));
        check(cudaMalloc(reinterpret_cast<void**>(&r.dst), output.size()));
        check(cudaEventCreate(&r.start));
        check(cudaEventCreate(&r.stop));
        check(cudaMemset(r.dst, 0xA5, output.size()));
        check(cudaMemcpy(r.src, input.data(), input.size(), cudaMemcpyHostToDevice));
        const dim3 block(bx, by), grid((width + bx - 1) / bx, (height + by - 1) / by);
        auto launch = [&]() {
            bgr_to_gray<<<grid, block>>>(r.src, r.dst, width, height, src_stride, dst_stride);
            check(cudaGetLastError());
        };
        std::ofstream samples(args.at("--samples"));
        if (!samples) throw std::runtime_error("cannot write samples");
        samples << "mode,iteration,duration_ns\n";
        for (int i = -warmup; i < iterations; ++i) {
            check(cudaEventRecord(r.start));
            launch();
            check(cudaEventRecord(r.stop));
            check(cudaEventSynchronize(r.stop));
            float ms = 0;
            check(cudaEventElapsedTime(&ms, r.start, r.stop));
            if (i >= 0) samples << "kernel_event," << i << "," << static_cast<double>(ms) * 1e6 << "\n";
        }
        // Independent synchronous, pageable-memory H2D + launch + sync + D2H wall timing.
        // Device/host allocations and file I/O are excluded from both modes.
        for (int i = -warmup; i < iterations; ++i) {
            const auto start = std::chrono::steady_clock::now();
            check(cudaMemcpy(r.src, input.data(), input.size(), cudaMemcpyHostToDevice));
            launch();
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(output.data(), r.dst, output.size(), cudaMemcpyDeviceToHost));
            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (i >= 0) samples << "transfer_kernel_wall," << i << "," << elapsed << "\n";
        }
        for (int y = 0; y < height; ++y)
            for (size_t x = width; x < dst_stride; ++x)
                if (output[y * dst_stride + x] != 0xA5)
                    throw std::runtime_error("destination padding overwritten");
        std::ofstream out(args.at("--output"), std::ios::binary);
        for (int y = 0; y < height; ++y)
            out.write(reinterpret_cast<char*>(output.data() + y * dst_stride), width);
        if (!out || !samples) throw std::runtime_error("output write failed");
        std::cout << "device=" << properties.name << " compute=" << properties.major << "."
                  << properties.minor << " block=" << bx << "x" << by << " grid="
                  << grid.x << "x" << grid.y << " src_stride=" << src_stride
                  << " dst_stride=" << dst_stride << " padding=unchanged\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
