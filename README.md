# NMRV

This repository contains coursework for visual tracking and motion estimation, organized by one exercise and four projects.

## Repository structure

- `/home/runner/work/NMRV/NMRV/Exercise1` — **Exercise 1: Optical Flow**
- `/home/runner/work/NMRV/NMRV/Project2` — **Project 2: Mean-Shift Tracker**
- `/home/runner/work/NMRV/NMRV/Project3_` — **Project 3: Correlation Filter Tracking**
- `/home/runner/work/NMRV/NMRV/Project4` — **Project 4: Advanced Tracking (Kalman + Particle Filter)**
- `/home/runner/work/NMRV/NMRV/Project5` — **Project 5: Long-Term Tracking (SiamFC)**

Some Exercise 1 helper files also exist at repository root (`ex1_utils.py`, `of_methods.py`, `run_vaja_1.py`, `test_methods.py`).

## Exercise 1: Optical Flow

Folder: `/home/runner/work/NMRV/NMRV/Exercise1`

### What is implemented
- Lucas–Kanade optical flow (`lucas_kanade`)
- Horn–Schunck optical flow (`horn_schunck`)
- Pyramidal Lucas–Kanade (`pyramidal_lk`)
- Runtime and parameter comparison helpers

### Key files
- `of_methods.py` — core optical-flow algorithms
- `ex1_utils.py` — image derivatives, smoothing, and visualization helpers
- `test_methods.py` — experiments/plots used for analysis
- `run_vaja_1.py` — quick demo on synthetic rotated noise
- `report/main.pdf` — exercise report

## Project 2: Mean-Shift Tracker

Folder: `/home/runner/work/NMRV/NMRV/Project2`

### What is implemented
- Mean-shift mode seeking with multiple kernels and stopping criteria
- Histogram-based mean-shift tracker for VOT sequences
- Parameter analysis (bins, alpha, iterations)

### Key files
- `ms_mode.py` — mean-shift mode-seeking experiments
- `ms_tracker.py` — mean-shift tracker implementation
- `run_tracker.py` — tracker runner and failure/reinitialization logic
- `ncc_tracker_example.py` — baseline NCC tracker example
- `sequence_utils.py`, `ex2_utils.py` — dataset/utility support
- `main.pdf` — project report

## Project 3: Correlation Filter Tracking

Folder: `/home/runner/work/NMRV/NMRV/Project3_`

### What is implemented
- Simplified MOSSE-style correlation filter tracker
- Filter training/update in Fourier domain
- Parameter sweeps (alpha, sigma, etc.) and result logging

### Key files
- `corr_tracker.py` — correlation filter tracker implementation
- `run_tracker.py` — sequence evaluation loop
- `report_plot.py` — plotting utilities for experiment outputs
- `tests_/` — saved experiment results
- `report/main.pdf` — project report

## Project 4: Advanced Tracking

Folder: `/home/runner/work/NMRV/NMRV/Project4`

### What is implemented
- Kalman filtering with RW, NCV, and NCA motion models
- Particle filter trackers (RW/NCV/NCA variants)
- Evaluation on VOT sequences with overlap/failure/speed metrics

### Key files
- `kalman_exp.py` — Kalman model experiments on synthetic trajectories
- `particle_tracker.py` — NCV particle tracker
- `particle_tracker_RW.py`, `particle_tracker_NCA.py` — model variants
- `run_tracker.py`, `run_tracker_2.py` — single/all-sequence evaluation scripts
- `ex4_utils.py` — Kalman and Gaussian sampling helpers
- `report/main.pdf` — project report

## Project 5: Long-Term Tracking

Folder: `/home/runner/work/NMRV/NMRV/Project5`

### What is implemented
- SiamFC-based tracker integration
- Long-term tracking adjustments (confidence threshold + re-detection sampling)
- Precision/recall/F-score evaluation and visualization

### Key files
- `run_tracker.py` — runs tracker and stores bounding boxes/scores
- `performance_evaluation.py` — computes precision, recall, and F-score
- `show_tracking.py` — visualizes predictions vs. ground truth
- `siamfc/` — SiamFC model, transforms, losses, and tracker logic
- `tools/sequence_utils.py` — sequence I/O and result serialization
- `report/main.pdf` — project report

## Notes for running experiments

- Most scripts assume a local VOT-style dataset path (commonly set as `vot14` or via CLI arguments).
- Many runner scripts open OpenCV windows for interactive visualization.
- Project 5 requires PyTorch and a SiamFC network weights file (e.g., `siamfc_net.pth`).
