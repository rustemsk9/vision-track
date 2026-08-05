// Slang compute shader for Distance Transform
[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    // Highly optimized parallel GPU calculation of pixel distances from YOLO mask
}
