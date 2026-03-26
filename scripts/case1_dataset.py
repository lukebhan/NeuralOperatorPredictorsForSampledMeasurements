"""Generate the training dataset for Case 1: sampling-horizon prediction operator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case1_dataset_builder import (
    build_predictor_dataset_parallel,
    save_predictor_dataset,
    validate_dataset_shapes,
    validate_dataset_labels,
)
from src.config import load_config
from src.simulate import build_robot, make_reference, make_simulator

cfg = load_config()


dataset = build_predictor_dataset_parallel(
    cfg,
    n_rollouts=20,
    stride=20,
    seed=0,
    max_workers=10,
)

save_predictor_dataset(dataset, cfg, "dataset/singlestep_predictor_dataset_small.npz")
robot = build_robot(cfg["urdf"])
ref = make_reference(robot, cfg)
sim = make_simulator(robot, cfg, ref)
validate_dataset_shapes(dataset, robot, cfg)
validate_dataset_labels(dataset, sim, cfg)