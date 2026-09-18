from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a LLaMA-Factory QLoRA experiment")
    parser.add_argument("--config", default="training/configs/qwen35_9b_qlora_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without training")
    args = parser.parse_args()

    config = (ROOT / args.config).resolve()
    if not config.exists():
        raise FileNotFoundError(config)
    command = ["llamafactory-cli", "train", str(config)]
    print("Command:", " ".join(command))
    if args.dry_run:
        return
    if shutil.which(command[0]) is None:
        raise RuntimeError("llamafactory-cli is not installed; use requirements-train.txt")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

