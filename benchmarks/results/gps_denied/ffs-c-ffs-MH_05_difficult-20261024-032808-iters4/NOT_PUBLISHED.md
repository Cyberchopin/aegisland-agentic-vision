# Not published: dataset camera imagery

The following files are listed in this directory's SHA256SUMS but are not included in this public branch, because each image contains raw EuRoC camera frames (cam0 left image, alone or with a depth/disparity overlay):

- `vis-000-left-ffs-sgbm.jpg`
- `vis-020-left-ffs-sgbm.jpg`

They are kept locally, unchanged, in the original working copy at `~/aegisland-agentic-vision/benchmarks/results/gps_denied/ffs-c-ffs-MH_05_difficult-20261024-032808-iters4/`, and in commit history of the private local branch `work/dronehacks-gps-denied-c`. SHA256SUMS is left exactly as generated, so a local copy can still be verified with `sha256sum -c SHA256SUMS`; in this branch those lines will report the files as missing.

The remaining files in this directory are numeric results without imagery.
