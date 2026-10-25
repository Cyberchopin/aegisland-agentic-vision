# Questions for DroneHacks organizers

Contact listed on <https://dronehacks.groups.stanford.edu/> (checked 2026-10-24): hello@stanfordhorizons.com.
The page confirms: Nov 6-8 2026, Stanford Robotics Center, 7" kits, Pixhawk 6C, "Luxonis
depth cameras", "NVIDIA Orin Nanos", prizes (Grand, Runner-Up, Third, Most Technologically
Impressive, Fastest Run). It does **not** state anything below.

| # | Question | Why it matters | Blocks |
| --- | --- | --- | --- |
| 1 | Exact Luxonis camera model (e.g. OAK-D Lite / Pro / S2 / W), firmware, and how it is mounted (forward / down / angle) | Stereo baseline, FOV, rolling vs global shutter, whether cuVSLAM stereo is viable; mounting sets extrinsics | 1 (field transfer), 2, 3 |
| 2 | Does that camera have an IMU (which chip, rate), and is it hardware-timestamped against the image sensors? | Stereo-inertial only works with a calibrated, time-aligned IMU; otherwise use Pixhawk IMU with clock sync | 1, 3 |
| 3 | Orin Nano variant (4 GB / 8 GB, Super or not), JetPack / L4T version, CUDA version, preinstalled software, root access | Chooses cuVSLAM wheel (JetPack 6.x → cu12 `cp310 aarch64`), memory budget for depth models | 1 (deploy), 2 |
| 4 | Can we flash / install our own packages and Python wheels? Internet on the Orin at the venue? | Offline install bundle vs on-site install | 1-4 |
| 5 | Pixhawk 6C firmware: PX4 or ArduPilot, version; is external vision input (`VISION_POSITION_ESTIMATE` / `ODOMETRY`) allowed and pre-configured? | Message format and EKF2 setup for stage 3; we stay read-only until allowed | 3 |
| 6 | Companion link: Orin ↔ Pixhawk connection (UART / USB, baud), MAVLink router, ROS 2 present or not | Interface design for shadow supervision | 3 |
| 7 | Are pre-written code and open-source libraries allowed? Must code be written during the event? Must it be open-sourced? | Whether this harness / cuVSLAM may be used at all | all |
| 8 | Scoring rubric and the actual task (course, distance, obstacles, landing target, time limit); how "Fastest Run" is measured | What accuracy / latency matters; what to optimize | 1-4 |
| 9 | Arena: indoor? lighting, texture, reflective floor, netting, size; any motion-capture ground truth for practice? | Visual odometry failure modes; ability to measure real ATE | 1, 3 |
| 10 | Safety rules: who is allowed to arm, kill switch, tether, max altitude/speed, required manual pilot, when we may first fly | Defines what "flight test" means and when SITL must suffice | 3, 4 |
| 11 | Will we get kit access or sample sensor recordings before the event (bag files / raw stereo+IMU)? | Lets us calibrate and replay before Nov 6 | 1, 2 |
| 12 | Team size and whether all registered members must be present | Work split | planning |
| 13 | Ground station / network: Wi-Fi provided? radio telemetry? laptops allowed to compute off-board? | Whether off-board GPU is legal or only Orin | 2, 3 |
| 14 | Calibration: are camera intrinsics/extrinsics and camera-IMU calibration provided, or calibration targets available on site? | Required input for any metric VIO | 1, 3 |

Do not assume answers from the OpenCV-2026 competition; its rules do not apply here.
