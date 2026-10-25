"""
Glue between AegisLand EuRoC recordings and Kalibr (calibration doc, steps 3 and 6).

  to-kalibr   <recording>/mav0 -> folder for `kalibr_bagcreater --folder`
              (cam0/<ts>.png, cam1/<ts>.png as symlinks, imu0.csv)
  from-kalibr camchain-imucam.yaml + imu.yaml -> NEW recording copy whose
              sensor yaml files come from Kalibr and whose IMU stamps are
              corrected by the estimated time shift (t_imu - shift); images are
              hard-linked, nothing in the source recording is modified

Run with ~/.venvs/depthai (numpy, PyYAML) and PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

from aegisland.euroc_writer import write_calibration
from aegisland.kalibr_import import from_kalibr


def to_kalibr(mav0: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    for camera in ("cam0", "cam1"):
        (out / camera).mkdir()
        for image in sorted((mav0 / camera / "data").glob("*.png")):
            (out / camera / image.name).symlink_to(image.resolve())
    shutil.copyfile(mav0 / "imu0" / "data.csv", out / "imu0.csv")
    print(f"kalibr_bagcreater --folder {out} --output-bag {out}.bag")


def from_kalibr_cmd(mav0: Path, camchain: Path, imu: Path, out: Path, rate: float) -> None:
    calibration = from_kalibr(yaml.safe_load(camchain.read_text()),
                              yaml.safe_load(imu.read_text()), camera_rate_hz=rate)
    target = out / "mav0"
    target.mkdir(parents=True, exist_ok=False)
    for camera in ("cam0", "cam1"):
        (target / camera / "data").mkdir(parents=True)
        shutil.copyfile(mav0 / camera / "data.csv", target / camera / "data.csv")
        for image in (mav0 / camera / "data").glob("*.png"):
            os.link(image, target / camera / "data" / image.name)
    (target / "imu0").mkdir()
    shift_ns = round(calibration.timeshift_cam_imu_s * 1e9)
    lines = (mav0 / "imu0" / "data.csv").read_text().splitlines()
    corrected = [lines[0]] + [f"{int(row.split(',', 1)[0]) - shift_ns},{row.split(',', 1)[1]}"
                              for row in lines[1:] if row]
    (target / "imu0" / "data.csv").write_text("\n".join(corrected) + "\n")
    files = write_calibration(target, calibration.cam0, calibration.cam1,
                              cam0_from_cam1=calibration.cam0_from_cam1,
                              cam0_from_imu=calibration.cam0_from_imu, imu=calibration.imu,
                              source=f"Kalibr {camchain.name}")
    (out / "calibration_source.json").write_text(json.dumps({
        "source_recording": str(mav0), "camchain": str(camchain), "imu": str(imu),
        "timeshift_cam_imu_s": calibration.timeshift_cam_imu_s,
        "imu_stamps_shifted_ns": -shift_ns, "files": files}, indent=1))
    print(json.dumps({"written": files, "imu_shift_ns": -shift_ns}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("to-kalibr")
    a.add_argument("--mav0", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("from-kalibr")
    b.add_argument("--mav0", type=Path, required=True)
    b.add_argument("--camchain", type=Path, required=True, help="camchain-imucam.yaml")
    b.add_argument("--imu", type=Path, required=True, help="imu.yaml given to Kalibr")
    b.add_argument("--camera-rate-hz", type=float, default=20.0)
    b.add_argument("--output", type=Path, required=True, help="new recording root")
    args = parser.parse_args()
    if args.command == "to-kalibr":
        to_kalibr(args.mav0, args.output)
    else:
        from_kalibr_cmd(args.mav0, args.camchain, args.imu, args.output, args.camera_rate_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
