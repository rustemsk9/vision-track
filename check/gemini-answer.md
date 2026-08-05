Because this error only occurs in Safari, the root cause is almost certainly content encoding/decompression handling, HTTP Range requests, or strict CORS policies specific to WebKit (Safari's rendering engine).When Safari encounters certain server headers (like gzip/br compression misconfigurations or partial range request headers on large files), it drops or truncates the binary payload, causing ONNX Runtime Web to parse incomplete data and throw protobuf parsing failed.Key Causes & Fixes for Safari1. Incomplete Response Handling / Range RequestsSafari often issues Range: bytes=0- headers when fetching large binary files (.onnx). If Streamlit's internal static file web server returns a 206 Partial Content status or handles range requests improperly, Safari passes an incomplete ArrayBuffer to ort.js.Fix: Explicitly request the full file using an explicit cache and headers configuration in fetch:JavaScriptconst response = await fetch(modelUrl, {
  method: 'GET',
  headers: {
    'Accept': 'application/octet-stream',
    'Range': 'bytes=0-' // Force standard fetch behavior on WebKit
  },
  cache: 'force-cache'
});

const arrayBuffer = await response.arrayBuffer();

// Verify byte length is non-zero before loading
if (arrayBuffer.byteLength < 1000) {
  throw new Error(`File received is too small (${arrayBuffer.byteLength} bytes). Safari likely truncated the request.`);
}

session = await ort.InferenceSession.create(arrayBuffer, { executionProviders: ['wasm'] });
2. Gzip / Content-Encoding Compression IssuesSafari automatically decompresses .onnx files if Streamlit or a proxy (like Nginx, Cloudflare, or AWS) serves them with Content-Encoding: gzip or Content-Encoding: br. If the server sends compressed bytes without the proper header, Chrome tolerates it or detects the binary, but Safari fails silently and outputs corrupt raw bytes.Fix (in Streamlit / Server): Ensure static .onnx files are excluded from gzip/brotli compression on your host or reverse proxy.Fix (Client Workaround): Explicitly download the model as a Blob first, convert it to an ArrayBuffer, and feed it to ort.js:JavaScriptconst response = await fetch(modelUrl);
const blob = await response.blob();
const arrayBuffer = await blob.arrayBuffer();

session = await ort.InferenceSession.create(arrayBuffer, { executionProviders: ['wasm'] });
3. Safari WebAssembly & COOP/COEP Isolation RestrictionsSafari enforces strict Cross-Origin-Opener-Policy (COOP) and Cross-Origin-Embedder-Policy (COEP) rules for WebAssembly and SharedArrayBuffer memory allocations.Fix: Tell ONNX Runtime Web to explicitly disable thread pooling / shared memory if you haven't enabled COOP/COEP headers on Streamlit:JavaScript// Add this BEFORE creating the InferenceSession
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false; // Disable SIMD if running on older iOS/Safari versions

session = await ort.InferenceSession.create(modelUrl, { 
    executionProviders: ['wasm'] 
});
How to Debug in SafariOpen Safari $\rightarrow$ Settings $\rightarrow$ Advanced $\rightarrow$ Check "Show features for web developers".Open your Streamlit app in Safari, open Develop $\rightarrow$ Show Web Inspector $\rightarrow$ Network tab.Click on your .onnx request and inspect:Transfer Size vs. Size: If Transfer Size is tiny (e.g., 200 B) compared to actual size (e.g., 20 MB), Safari was served a truncated stream.Headers: Check Content-Type. It should ideally be application/octet-stream or application/wasm.