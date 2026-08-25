"""
Script: scripts/run_inference.py

Purpose:
CLI entrypoint for production-grade clinical inference on BraTS studies.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import ClinicalInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="FedMed Clinical Inference Runner")
    parser.add_argument("--subject-dir", type=str, required=True, help="Path to patient subject directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fedavg/global_best.pt", help="Model checkpoint")
    parser.add_argument("--config", type=str, default="configs/production/inference.yaml", help="Inference config")
    parser.add_argument("--output", type=str, default="reports/inference_output.nii.gz", help="Output segmentation path")
    args = parser.parse_args()

    s_dir = Path(args.subject_dir)
    mod_paths = {}
    for mod in ("t1", "t1ce", "t2", "flair"):
        matches = sorted(list(s_dir.glob(f"*{mod}.nii*")))
        if matches:
            mod_paths[mod] = matches[0]

    engine = ClinicalInferenceEngine(
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
    )

    res = engine.predict_subject(mod_paths, output_nifti_path=Path(args.output))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
