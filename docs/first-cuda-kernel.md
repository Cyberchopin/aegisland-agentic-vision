# First CUDA kernel: measured, not projected

This supersedes the **plan-only** status of the BGR8→GRAY8 experiment in
`cpu-profiling.md`. Production perception is unchanged. This is one standalone
kernel and a correctness experiment, not an integrated GPU perception pipeline.
No tiling, fusion, asynchronous overlap, Canny/Farneback rewrite or Stage 5+ work.

## Build and run

Windows on this machine (RTX 4060 Ti / CC 8.9, CUDA 12.8, VS 2019 Build Tools):

```powershell
cmd /c cpp\cuda_gray\build_windows.cmd
.venv/Scripts/python.exe -m aegisland.cuda_gray --exe work/cuda-build/gray_bench.exe --input ../../outputs/cpu-stage0-isolated/inputs.npz --output ../../outputs/cuda-gray-validation --warmup 10 --iterations 100
$env:AEGISLAND_GRAY_EXE = (Resolve-Path work/cuda-build/gray_bench.exe).Path
.venv/Scripts/python.exe -m pytest tests/test_cuda_gray.py tests/test_profiling.py --basetemp=work/pytest-cuda
.venv/Scripts/python.exe -m ruff check src tests
```

Use a fresh output directory for each run. `build_windows.cmd` initializes the
installed compiler environment within the child process; it does not change PATH
globally. Linux build equivalent, with a compatible host compiler and the same GPU:

```sh
mkdir -p work/cuda-build
nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 -Xptxas=-v cpp/cuda_gray/gray_bench.cu -o work/cuda-build/gray_bench
```

The Linux command has not been run here. WSL distribution enumeration was denied
in this sandbox, so no environment migration was performed. Both nvcc and ncu
were already installed on Windows; no CUDA Python binding or extra dependency was needed.

## Implementation and actual correctness

One thread processes one pixel with a 2D 32×8 block, explicit bounds checks and
source/destination byte strides. BGR8 values use the integer conversion
`(3735*B + 19235*G + 9798*R + 16384) >> 15`. This follows the 15-bit OpenCV
integer convention and was tested against **installed OpenCV 5.0.0**, including
every possible BGR8 value. It is not an assumption of floating-point equivalence.
Reference: [OpenCV conversion source](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/color_rgb.simd.hpp).

The run tested 18 cases / 17,929,593 pixels:

- 14 existing synthetic replay frames at 320×240.
- All 16,777,216 BGR8 triplets arranged in a 4096×4096 image.
- Seed-42 random 1×1, 33×9 and 319×241 images with padded source rows.

All cases have **maximum absolute error 0, MAE 0, mismatch fraction 0** against
cv2.cvtColor. Signed error histograms are saved per case. The strict tolerance
remains zero; it was not relaxed to make a failing implementation pass. Exhaustive
color validation does not prove all possible memory layouts, so bounds/stride cases
and a Compute Sanitizer memcheck were also run. Memcheck reported **0 errors**.
Destination rows have 13 sentinel bytes; the driver verifies they remain unchanged.

The actual GPU gray outputs were then substituted into a separate CPU replay at
the production BGR→gray call. All downstream arrays and public evidence (excluding
processing_ms) passed the existing strict comparison gate. This uses cached GPU
results solely for correctness propagation, **not** for measuring a GPU pipeline.
Active CLAHE retries are not claimed as GPU-accelerated by this experiment.

## Actual Nsight Compute collection

The first unprivileged run launched the kernel but returned ERR_NVGPUCTRPERM.
The user subsequently ran the prepared script as administrator. The resulting
report contains a real bgr_to_gray launch and **8 profiler replay passes**.
No driver permission policy was changed by the script.
The installed ncu help confirms the defaults used: replay-mode=kernel,
cache-control=all and clock-control=base. The latter requests base-clock locking
during profiling; it is distinct from the unprofiled loops, which set no clocks.
The report records the observed SM frequency as 2.26 GHz and DRAM frequency as 8.69 GHz.

```powershell
ncu --kernel-name-base function --kernel-name bgr_to_gray --launch-skip 10 --launch-count 1 --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section SpeedOfLight --export ../../outputs/cuda-gray-validation/first-gray-admin work/cuda-build/gray_bench.exe --input ../../outputs/cuda-gray-validation/frame-000.bgr --output ../../outputs/cuda-gray-validation/ncu-admin.gray --samples ../../outputs/cuda-gray-validation/ncu-admin-timings.csv --width 320 --height 240 --warmup 10 --iterations 20
ncu --import ../../outputs/cuda-gray-validation/first-gray-admin.ncu-rep --page raw --csv
ncu --import ../../outputs/cuda-gray-validation/first-gray-admin.ncu-rep --page details
```

RTX 4060 Ti, driver 595.95, Nsight Compute 2025.1.0, 320×240 input,
32×8 block, 10×30 grid / 300 blocks, 34 SMs, 1.47 waves/SM:

| Metric as exported by installed ncu | Measured value |
| --- | ---: |
| sm__warps_active.avg.pct_of_peak_sustained_active | 72.792803% |
| sm__maximum_warps_per_active_cycle_pct | 100% theoretical |
| gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed | 32.368608% |
| dram__bytes.sum.per_second | 90.024691 Gbyte/s |
| gpu__time_duration.sum | 2.592 microseconds |
| launch__registers_per_thread | 16 |
| launch__occupancy_limit_blocks | 24 blocks |
| launch__occupancy_limit_registers | 16 blocks |
| launch__occupancy_limit_shared_mem | 16 blocks |
| launch__occupancy_limit_warps | 6 blocks |

The installed version exports **gpu__dram_throughput**, not a fabricated value
for an unavailable dram__throughput alias. The original CSV, units row and ncu-rep
are retained. The compiler build log separately reports 13 registers and no spills;
the table above quotes Nsight's launch report (16), without conflating the two.

This launch did **not** saturate DRAM bandwidth. Nsight flags low compute/memory
utilization and a partial final wave. Those are clues, not proof of a unique
bottleneck. Its rule-based estimated speedups are not measured improvements and
are not claimed here. One very short launch collected across replay passes cannot
establish stable throughput or a general memory-bandwidth-bound classification.

## Timing boundaries and limits

The driver records two independent loops with 10 warmups and 100 measured
iterations per normal replay frame:

1. `kernel_event`: CUDA events around a launch using already resident device input;
   synchronize the stop event before reading duration.
2. `transfer_kernel_wall`: steady_clock around synchronous pageable-memory H2D,
   launch, device synchronization and D2H. No CUDA events in this loop.

Both exclude allocation, context setup and file I/O. Transfer copies include source
and destination row padding. Default-stream execution is intentional; no overlap.
The CPU primitive comparison uses perf_counter_ns around cv2.cvtColor with its
normal allocation behavior. Neither primitive measurement is whole observe latency.

The initial measurements overlapped the CPU regression suite and ran on a desktop
display GPU with no locked clocks. In that preliminary run, frame-000 kernel-event
p50 was 4.000 microseconds; transfer+kernel wall p50 was 74.750 microseconds.
These are raw observations, not a speedup claim. Keep them separate from Nsight's
2.592-microsecond profiler duration and from any later isolated rerun.

## Where the existing CPU data points

From the earlier independent CPU diagnostic samples (14 synthetic frames × 20 runs):
observe inclusive time summed to 8.3008858 s; Farneback calls summed to 4.6866991 s;
camera compensation to 2.3965816 s; cvtColor to 0.0087665 s. These are **instrumented
diagnostic totals**, not production end-to-end timings, and parent/child totals overlap.
They put Farneback at about 56.5% and compensation at about 28.9% of observe, while
gray conversion is about 0.1%. This supports investigating optical flow next on
representative data, rather than optimizing gray conversion. No new kernel is started here.

## Test-suite investigation

The verbose full-suite run records progress and requests Python stack dumps after
60 seconds inside a test. The suite contains repeated full scenario simulations,
not just unit tests: three fault-timeline tests each regenerate an 80-frame scenario;
ablation tests repeatedly regenerate multiple scenarios for separate assertions.

A separate cProfile of run_fault_timeline (80 frames, 96 observe calls including
retries) measured **190 Farneback calls**. With default 28 OpenCV threads it took
22.259 s, of which 18.610 s was Farneback. With one thread it took 20.830 s,
of which 17.247 s was Farneback. These diagnostic runs overlapped the full suite;
they identify where time went, not a reliable thread-count speedup. They provide
positive evidence of expensive repeated computation, not evidence of a network hang.
The 60-second stack dump for
`test_safety_assessment.py::test_deterministic_safety_gates_pass` landed in
`perception.py::_motion` at the raw Farneback call, through
`build_assessment → run_ablation → run_fault_timeline → agent.step → observe`.
That test subsequently passed. A stack-dump timeout is diagnostic output, not a
test failure or evidence that the test was killed.
The final full-suite status and durations are recorded with the execution report.
The complete run finished: **201 passed in 1188.23 seconds (19:48)**. The four
slowest tests were the safety-assessment tests, each taking 98.36–98.66 seconds.
The three newly added CUDA tests were run separately and also passed (the real
GPU integration test was enabled). A pytest cache-directory write warning did not
affect assertions. No tests were skipped to complete the 201-test run.

After the full suite finished, a separate unprofiled rerun again passed all 18
cases. For frame-000, 100 samples after 10 warmups gave p50 values of 4.096 us
(kernel event), 74.850 us (H2D/kernel/sync/D2H wall), and 17.400 us (CPU cvtColor
wall). These are primitive timings with the boundaries above; they do not show
a whole-pipeline GPU speedup. The agent's tests no longer overlapped that rerun,
but other desktop applications and clocks remained uncontrolled.

## Artifacts

`outputs/cuda-gray-validation/` retains per-case BGR inputs, actual gray outputs,
CPU reference bytes, correctness/error histograms, native command arguments and
logs, CPU/GPU CSV samples, environment/build provenance, and the Nsight report.
`outputs/test-diagnosis/` retains the diagnostic CPU profiles and JSON summaries.
Use the execution report for the absolute workspace paths and completed-test status.

References: [NVIDIA metric interpretation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
[CLI and replay behavior](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html),
[counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM).
