"""Generate the training dataset for Case 2: predictor approximation operator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case2_dataset_builder import (
    build_multistep_predictor_dataset_parallel,
    save_multistep_predictor_dataset,
    validate_multistep_dataset_labels,
    validate_multistep_dataset_shapes,
)
from src.simulate import build_robot, make_reference, make_simulator
from src.config import load_config

cfg = load_config()

dataset = build_multistep_predictor_dataset_parallel(
    cfg,
    n_rollouts=20,
    stride=20,
    seed=0,
    max_workers=10,
)

save_multistep_predictor_dataset(dataset, cfg, "dataset/multistep_predictor_dataset_small.npz")
robot = build_robot(cfg["urdf"])
ref = make_reference(robot, cfg)
sim = make_simulator(robot, cfg, ref)
validate_multistep_dataset_shapes(dataset, robot, cfg)
validate_multistep_dataset_labels(dataset, sim, cfg)
