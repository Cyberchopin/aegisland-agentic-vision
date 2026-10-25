"""
Record a Luxonis OAK (DepthAI v3) device straight into the EuRoC/ASL layout.

Records:
  - raw left/right mono frames, which become cam0 and cam1;
  - on-device stereo depth, optional, which becomes depth0 (uint16 mm);
  - IMU gyro + accel, merged onto the gyro stamps, which becomes imu0;
  - the factory calibration from EEPROM, which becomes the sensor yaml files.
All of it is written with aegisland.euroc_writer, so run_cuvslam_euroc.py, the
diagnostics and the supervisor replay work on the output unchanged.

Timestamps:
  - every stamp is the DEVICE clock (`getTimestampDevice()`), in integer ns;
  - the host-synced stamp of each frame is kept in host_timestamps.csv, for
    latency analysis only.

Model-dependent parts. These are checked at runtime and recorded in
recording.json; they are not assumed:
  - IMU present? Some OAK-D Lite batches ship without one. Without an IMU
    there is no imu0, so the recording is stereo-only.
  - Mono sensor and resolution:
    - OV7251 (OAK-D Lite) 640x480;
    - OV9282 (OAK-D / Pro / S2 / W) 1280x800, or 640x400 binned.
  - Shutter: global on both of those. Note the colour camera is NOT used.
  - IR dot projector (Pro models only): forced off. The pattern moves with the
    camera and creates false features.
  - Lens model:
    - perspective boards give 14 coefficients, and only k1,k2,p1,p2 are kept
      (radtan, sensor.yaml) — a loss, so recalibrate with Kalibr;
    - fisheye (W) boards give equidistant k1..k4 (sensor_recalibrated.yaml).
  - Factory extrinsics: DepthAI returns centimetres by default; METER is
    requested explicitly. The direction of the IMU<->camera transforms must be
    verified against a Kalibr calibration (see docs/dronehacks/camera-imu-calibration.md).

Needs a connected device. Without one it exits with a clear error and writes nothing.
Run with ~/.venvs/depthai and PYTHONPATH=src. Writes a NEW directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aegisland.euroc_writer import CameraCalibration, EurocRecordingWriter, ImuNoise, merge_imu

# Datasheet-order defaults for the IMU noise model when EEPROM has none; replace with
# Allan-variance values (calibration doc §4) before trusting stereo-inertial results.
DEFAULT_IMU_NOISE = ImuNoise(1.7e-4, 1.9e-5, 2.0e-3, 3.0e-3, 200.0)


def device_ns(message) -> int:
    return int(message.getTimestampDevice().total_seconds() * 1e9)


def host_ns(message) -> int:
    return int(message.getTimestamp().total_seconds() * 1e9)


def build_pipeline(dai, args):
    pipeline = dai.Pipeline()
    device = pipeline.getDefaultDevice()
    sockets = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
    cameras = [pipeline.create(dai.node.Camera).build(socket) for socket in sockets]
    outputs = [camera.requestOutput(tuple(args.resolution), type=dai.ImgFrame.Type.GRAY8,
                                    fps=args.fps) for camera in cameras]
    queues = {"left": outputs[0].createOutputQueue(maxSize=64, blocking=True),
              "right": outputs[1].createOutputQueue(maxSize=64, blocking=True)}
    if args.depth:
        stereo = pipeline.create(dai.node.StereoDepth).build(outputs[0], outputs[1])
        queues["depth"] = stereo.depth.createOutputQueue(maxSize=64, blocking=False)
    imu_type = device.getConnectedIMU()
    has_imu = bool(imu_type) and imu_type.upper() != "NONE" and not args.no_imu
    if has_imu:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                            args.imu_rate)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(20)
        queues["imu"] = imu.out.createOutputQueue(maxSize=256, blocking=False)
    return pipeline, device, queues, {"imu_type": imu_type, "has_imu": has_imu}


def calibration(dai, device, args, has_imu: bool):
    handler = device.readCalibration()
    left, right = dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C
    width, height = args.resolution
    cams = []
    for socket in (left, right):
        k = np.array(handler.getCameraIntrinsics(socket, width, height))
        coefficients = list(handler.getDistortionCoefficients(socket))
        fisheye = handler.getDistortionModel(socket) == dai.CameraModel.Fisheye
        model = "fisheye" if fisheye else "radtan"
        cams.append(CameraCalibration((width, height), (k[0, 0], k[1, 1], k[0, 2], k[1, 2]),
                                      model, tuple(coefficients[:4]), args.fps))
    meter = dai.LengthUnit.METER
    # getCameraExtrinsics(src, dst): maps points from src to dst = dst_from_src.
    cam0_from_cam1 = np.array(handler.getCameraExtrinsics(right, left, unit=meter))
    cam0_from_imu = (np.array(handler.getImuToCameraExtrinsics(left, unit=meter))
                     if has_imu else np.eye(4))
    return cams, cam0_from_cam1, cam0_from_imu, {
        "distortion_coefficients_full": {
            "cam0": list(handler.getDistortionCoefficients(left)),
            "cam1": list(handler.getDistortionCoefficients(right))},
        "cam0_from_cam1_m": cam0_from_cam1.tolist(), "cam0_from_imu_m": cam0_from_imu.tolist(),
        "note": "factory EEPROM calibration; verify with Kalibr before benchmarking",
    }


def record(args) -> int:
    try:
        import depthai as dai
    except ImportError:
        print("depthai is not installed in this environment", file=sys.stderr)
        return 2
    if not dai.Device.getAllAvailableDevices():
        print("no OAK device found; nothing written", file=sys.stderr)
        return 3
    pipeline, device, queues, imu_info = build_pipeline(dai, args)
    writer = EurocRecordingWriter(args.output, depth=args.depth)
    cams, cam0_from_cam1, cam0_from_imu, calib_info = calibration(dai, device, args,
                                                                  imu_info["has_imu"])
    writer.write_calibration(cams[0], cams[1], cam0_from_cam1=cam0_from_cam1,
                             cam0_from_imu=cam0_from_imu, imu=DEFAULT_IMU_NOISE,
                             source=f"OAK {device.getDeviceName()} factory calibration")
    projector = "not present"
    if device.getIrDrivers():
        device.setIrLaserDotProjectorIntensity(0.0)
        device.setIrFloodLightIntensity(0.0)
        projector = "forced off"
    host_rows, exposure_rows, gyro, accel = [], [], [], []
    pending_right: dict[int, object] = {}
    pipeline.start()
    started = time.monotonic()
    try:
        while pipeline.isRunning() and time.monotonic() - started < args.duration_s:
            left = queues["left"].get()
            right = queues["right"].get()
            pending_right[right.getSequenceNum()] = right
            match = pending_right.pop(left.getSequenceNum(), None)
            if match is not None:
                ok = writer.add_stereo(device_ns(left), left.getCvFrame(), device_ns(match),
                                       match.getCvFrame())
                if ok:
                    exposure_us = left.getExposureTime().total_seconds() * 1e6
                    host_rows.append(f"{device_ns(left)},{host_ns(left)}\n")
                    exposure_rows.append(f"{device_ns(left)},{exposure_us:.1f},"
                                         f"{left.getSensitivity()}\n")
            if "depth" in queues and (depth := queues["depth"].tryGet()) is not None:
                writer.add_depth(device_ns(depth), depth.getFrame().astype(np.uint16))
            if "imu" in queues:
                for packet_group in queues["imu"].tryGetAll():
                    for packet in packet_group.packets:
                        g, a = packet.gyroscope, packet.acceleroMeter
                        gyro.append((device_ns(g), g.x, g.y, g.z))
                        accel.append((device_ns(a), a.x, a.y, a.z))
    finally:
        pipeline.stop()
        # Always finalize what was captured, also after an error or Ctrl-C.
        finish(args, writer, device, dai, imu_info, projector, calib_info, host_rows,
               exposure_rows, gyro, accel)
    return 0


def finish(args, writer, device, dai, imu_info, projector, calib_info, host_rows,
           exposure_rows, gyro, accel) -> None:
    for row in merge_imu(np.array(gyro or np.empty((0, 4))), np.array(accel or np.empty((0, 4)))):
        writer.add_imu(int(row[0]), row[1:4], row[4:7])
    (args.output / "mav0" / "host_timestamps.csv").write_text(
        "#device_ns,host_synced_ns\n" + "".join(host_rows))
    (args.output / "mav0" / "exposure.csv").write_text(
        "#timestamp [ns],exposure_us,iso\n" + "".join(exposure_rows))
    manifest = writer.close({
        "device": {"name": device.getDeviceName(), "id": device.getDeviceId(),
                   "product": device.getProductName() if hasattr(device, "getProductName") else None,
                   "depthai": dai.__version__},
        "imu": imu_info, "ir_projector": projector, "resolution": args.resolution,
        "fps": args.fps, "calibration": calib_info,
        "imu_noise": "datasheet-order defaults, NOT measured",
    })
    print(json.dumps(manifest["counts"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, required=True, help="new directory")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--resolution", type=int, nargs=2, default=(640, 400),
                        metavar=("W", "H"), help="OV9282: 640 400 or 1280 800; OV7251: 640 480")
    parser.add_argument("--imu-rate", type=int, default=200)
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--depth", action="store_true", help="also record on-device depth")
    return record(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
