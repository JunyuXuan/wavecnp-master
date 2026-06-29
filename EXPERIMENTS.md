# Experiment Tracking

Use this document as the single experiment log for `wavecnp-git`. Each experiment
should be reproducible from the information here: where it ran, why it ran, the
exact command, what changed, and where the outputs live.

## Server Inventory

Update this section whenever a new server, GPU, path, or environment is used.

| Server alias | Hostname / IP | GPU / CUDA | Python env | Workspace path | Notes |
|---|---|---|---|---|---|
| `server-name` | `hostname` | `cuda:0`, GPU model, CUDA version | `conda/env-name` | `/home/juxuan/Data/workspace_py/wavecnp-git` | |

## Running Experiments

Keep one row per active run. Successful runs move to `Completed Experiments`;
failed or stopped runs move to `Incomplete Experiments`.

| Exp ID | Started | Server | Device | Aim | Command / config | Output path | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| `E012` | `2026-06-15 11:28` | `jarvis1.ihpc.uts.edu.au` | `cuda:0` | Run all 2D methods on selected image datasets | `RUN_ROOT="all_2d_methods_20260615_112828" METHODS="wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16` | `_experiments/all_2d_methods_20260615_112828; aggregate CSV in rsults/` | Running |  |
| `E006` | `2026-06-08 23:26` | `jarvis1.ihpc.uts.edu.au` | `cuda:0` | Run all 1D methods on selected GP kernels | `RUN_ROOT="all_1d_gp_kernels_20260608_232620" KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 10` | `_experiments/all_1d_gp_kernels_20260608_232620; aggregate CSV in rsults/` | Running |  |
| `E005` | `2026-06-07 21:30` | `jarvis1.ihpc.uts.edu.au` | `cuda:0` | Run all 1D methods on selected GP kernels | `KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20` | `_experiments/all_1d_gp_kernels_20260607_213023; aggregate CSV in rsults/` | Running |  |
| `E004` | `2026-06-07 14:49` | `jarvis1.ihpc.uts.edu.au` | `cuda:0` | Run all 1D methods on selected GP kernels to check the reults are well-formated or not | `KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 2` | `_experiments/all_1d_gp_kernels_20260607_144919; aggregate CSV in rsults/` | completed | the results look good |
| `E003` | `2026-06-07 14:49` | `jarvis1.ihpc.uts.edu.au` | `cuda:0` | Run all 1D methods on selected GP kernels | `KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20` | `_experiments/all_1d_gp_kernels_20260607_144911; aggregate CSV in rsults/` | stopped |  |
| `E002` | `2026-06-07 12:22` | `jarvis1.ihpc.uts.edu.au` screen `wave` | `cuda:0` | Run all 1D methods on selected GP kernels | `KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20` | `_experiments/all_1d_gp_kernels_20260607_122254; aggregate CSV in rsults/` | stopped |  |


## Completed Experiments

This table is the quick index. Put detailed settings, observations, and results
in `Experiment Log`.

| Exp ID | Completed | Server | Model | Task / data | Aim | Command / config | Result summary | Artifacts | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `E018` | `2026-06-21 18:22` | `jarvis1.ihpc.uts.edu.au` | `convcnp2 wavecnp2_improved_adapt wavecnp2_improved_noadapt` | `image / mnist` | `Compare ConvCNP2 with improved WaveCNP2 variants` | `RUN_ROOT="all_2d_methods_20260621_130104" METHODS="convcnp2 wavecnp2_improved_adapt wavecnp2_improved_noadapt" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:1" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200` | `` | `_experiments/all_2d_methods_20260621_130104; rsults/2d_all_methods_image_datasets_all_2d_methods_20260621_130104.csv` | `All requested method, dataset, and seed combinations completed successfully.` |
| `E015` | `2026-06-19 12:54` | `jarvis1.ihpc.uts.edu.au` | `wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp` | `image / mnist` | `Run all 2D methods on selected image datasets` | `RUN_ROOT="all_2d_methods_20260618_122814" METHODS="wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16` | `` | `_experiments/all_2d_methods_20260618_122814; rsults/2d_all_methods_image_datasets_all_2d_methods_20260618_122814.csv` | `All requested method, dataset, and seed combinations completed successfully.` |
| `E013` | `2026-06-17 09:51` | `jarvis1.ihpc.uts.edu.au` | `wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp` | `image / mnist` | `Run all 2D methods on selected image datasets` | `RUN_ROOT="all_2d_methods_20260615_212258" METHODS="wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16` | `40/40 MNIST runs aggregated from per-seed logs` | `_experiments/all_2d_methods_20260615_212258; rsults/2d_all_methods_image_datasets_all_2d_methods_20260615_212258.csv` | `All requested method, dataset, and seed combinations completed; aggregate CSV rebuilt from per-seed logs because summary.json files keep only the latest seed per method.` |
| `E011` | `2026-06-14 19:24` | `jarvis1.ihpc.uts.edu.au` | `wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp` | `image / mnist` | Run all 2D methods on selected image datasets |  | `_experiments/all_2d_methods_20260614_124852; rsults/2d_all_methods_image_datasets_all_2d_methods_20260614_124852.csv` | All requested method, dataset, and seed combinations completed successfully. |
| `E009` | `2026-06-14 11:52` | `jarvis1.ihpc.uts.edu.au` | `wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp` | `image / fake` | Smoke-test all 2D methods | All 8 methods completed for seed 1 | `_experiments/smoke_all_2d_methods_20260614; rsults/2d_all_methods_image_datasets_smoke_all_2d_methods_20260614.csv` | ANP failed once, then completed on retry. |
| `E000` | `2026-06-06` | `server-name` | `wavecnp` | `regression` / `matern` | Baseline comparison | `test_log_likelihood`: add mean +/- std | `rsults/1d_all_methods_matern_all_1d_matern_20260605_133111.csv` | |
| `E000` | `2026-06-06` | `server-name` | `convcnp` | `regression` / `matern` | Baseline comparison | `test_log_likelihood`: add mean +/- std | `rsults/1d_all_methods_matern_all_1d_matern_20260605_133111.csv` | |

## Incomplete Experiments

Runs listed here reached a terminal state without completing every requested
method, dataset, and seed combination.

| Exp ID | Started | Ended | Server | Aim | Status | Artifacts | Notes |
|---|---|---|---|---|---|---|---|
| `E019` | `2026-06-25 12:54` | `2026-06-27 13:21` | `jarvis1.ihpc.uts.edu.au` | Compare ConvCNP with 1D WaveCNP variants | Failed | `_experiments/all_1d_gp_kernels_20260625_125402; rsults/1d_all_methods_gp_kernels_all_1d_gp_kernels_20260625_125402.csv` | Runner exited with code 1; inspect the status CSV and logs. |
| `E017` | `2026-06-21 12:26` | `2026-06-21 17:26` | `jarvis1.ihpc.uts.edu.au` | Run all 2D methods on selected image datasets | Failed | `_experiments/all_2d_methods_20260621_122605; rsults/2d_all_methods_image_datasets_all_2d_methods_20260621_122605.csv` | Runner exited with code 2; inspect the status CSV and logs. |
| `E016` | `2026-06-21 12:25` | `2026-06-21 12:25` | `jarvis1.ihpc.uts.edu.au` | Run all 2D methods on selected image datasets | Stopped | `_experiments/all_2d_methods_20260621_122537; rsults/2d_all_methods_image_datasets_all_2d_methods_20260621_122537.csv` | Runner was interrupted; inspect the status CSV and logs before resuming. |
| `E014` | `2026-06-18 12:26` | `2026-06-18 12:27` | `jarvis1.ihpc.uts.edu.au` | Run all 2D methods on selected image datasets | Stopped | `_experiments/all_2d_methods_20260618_122657; rsults/2d_all_methods_image_datasets_all_2d_methods_20260618_122657.csv` | Runner was interrupted; inspect the status CSV and logs before resuming. |
| `E010` | `2026-06-14 12:29` | `2026-06-14 12:31` | `jarvis1.ihpc.uts.edu.au` | Run all 2D methods on MNIST | Stopped | `_experiments/all_2d_methods_20260614_122948; rsults/2d_all_methods_image_datasets_all_2d_methods_20260614_122948.csv` | Interrupted during WaveCNP2 seed 1 at epoch 23/50; no requested combination completed. |
| `E008` | `2026-06-12 17:15` | `2026-06-12 17:54` | `jarvis1.ihpc.uts.edu.au` | Run selected 2D methods on MNIST | Failed | `_experiments/all_2d_methods_20260612_171529; rsults/2d_all_methods_image_datasets_all_2d_methods_20260612_171529.csv` | 30/35 runs completed; LieCNP2 failed for all five seeds. |
| `E007` | `2026-06-09 20:05` | `2026-06-09 22:08` | `jarvis1.ihpc.uts.edu.au` | Run all 1D methods on selected GP kernels | Stopped | `_experiments/all_1d_gp_kernels_20260609_200505; rsults/1d_all_methods_gp_kernels_all_1d_gp_kernels_20260609_200505.csv` | 60/160 summaries exist: Matern completed, RBF partial, Polynomial and RBF+Linear missing. |

## Experiment Log

Copy this template for each experiment. Prefer keeping full commands here rather
than only linking to shell history, because shell history is easy to lose.

### E019 - Compare ConvCNP with 1D WaveCNP variants

**Status:** Failed

**Aim**

- Compare ConvCNP with 1D WaveCNP variants

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 1177191

**Experiment setup**

- Model: convcnp wavecnp wavecnpnew wavecnpimproved
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260625_125402

- Kernels: matern eq polynomial linear
- Seeds: 1 2 3 4 5
- Methods: convcnp wavecnp wavecnpnew wavecnpimproved
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_1d_gp_kernels_20260625_125402" METHODS="convcnp wavecnp wavecnpnew wavecnpimproved" KERNELS="matern eq polynomial linear" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 200
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260625_125402; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: Runner exited with code 1; inspect the status CSV and logs.

### E018 - Compare ConvCNP2 with improved WaveCNP2 variants

**Status:** Completed

**Aim**

- Compare ConvCNP2 with improved WaveCNP2 variants

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:1
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 12ac01d

**Experiment setup**

- Model: convcnp2 wavecnp2_improved_adapt wavecnp2_improved_noadapt
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260621_130104

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: convcnp2 wavecnp2_improved_adapt wavecnp2_improved_noadapt
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260621_130104" METHODS="convcnp2 wavecnp2_improved_adapt wavecnp2_improved_noadapt" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:1" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260621_130104; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: All requested method, dataset, and seed combinations completed successfully.

### E017 - Run all 2D methods on selected image datasets

**Status:** Failed

**Aim**

- compare wavecnp2 and convcnp2; wavecnp2 use level=4

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 12ac01d

**Experiment setup**

- Model: convcnp2 wavecnp2_adapt wavecnp2_noadapt
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260621_122605

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: convcnp2 wavecnp2_adapt wavecnp2_noadapt
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260621_122605" METHODS="convcnp2 wavecnp2_adapt wavecnp2_noadapt" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260621_122605; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: Runner exited with code 2; inspect the status CSV and logs.

### E016 - Run all 2D methods on selected image datasets

**Status:** Stopped

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 12ac01d

**Experiment setup**

- Model: convcnp2 wavecnp2_adapt wavecnp2_noadapt
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260621_122537

- Datasets: mnist
- Seeds: 1, 2, 3, 4, 5
- Methods: convcnp2 wavecnp2_adapt wavecnp2_noadapt
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260621_122537" METHODS="convcnp2 wavecnp2_adapt wavecnp2_noadapt" DATASETS="mnist" SEEDS="1, 2, 3, 4, 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260621_122537; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: Runner was interrupted; inspect the status CSV and logs before resuming.

### E015 - Run all 2D methods on selected image datasets

**Status:** Completed

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: efad894

**Experiment setup**

- Model: wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260618_122814

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260618_122814" METHODS="wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260618_122814; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: All requested method, dataset, and seed combinations completed successfully.
- The results look fine, but wavecnp2 is worese than convcnp2
- next, fix this issue. firsly try level=4 (original level=3)

### E014 - Run all 2D methods on selected image datasets

**Status:** Stopped

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: efad894

**Experiment setup**

- Model: wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260618_122657

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260618_122657" METHODS="wavecnp2_adapt wavecnp2_noadapt convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260618_122657; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: Runner was interrupted; inspect the status CSV and logs before resuming.

### E013 - Run all 2D methods on selected image datasets

**Status:** Completed

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: c06209d

**Experiment setup**

- Model: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260615_212258

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260615_212258" METHODS="wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260615_212258; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: All requested method, dataset, and seed combinations completed; aggregate CSV rebuilt from per-seed logs because summary.json files keep only the latest seed per method.

### E012 - Run all 2D methods on selected image datasets

**Status:** Running

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: c06209d

**Experiment setup**

- Model: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260615_112828

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260615_112828" METHODS="wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 200 --batch-size 16
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260615_112828; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 

### E011 - Run all 2D methods on selected image datasets

**Status:** Completed

**Aim**

- Run all 2D methods on selected image datasets

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: c06209d

**Experiment setup**

- Model: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260614_124852

- Datasets: mnist
- Seeds: 1 2 3 4 5
- Methods: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260614_124852" METHODS="wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 50
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260614_124852; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: All requested method, dataset, and seed combinations completed successfully.

### E010 - Run all 2D methods on MNIST

**Status:** Stopped

**Aim**

- Run all 2D methods on MNIST

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 7c056d8

**Experiment setup**

- Model: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260614_122948
- Datasets: mnist
- Seeds: 1 2 3 4 5
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260614_122948" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 50
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260614_122948; aggregate CSV in rsults/

**Results**

- Main metric: No completed test result
- Secondary metrics:
- Runtime: About 2 minutes
- Peak GPU memory:

**Observations**

- Notes: Interrupted during WaveCNP2 seed 1 at epoch 23/50; no requested combination completed.

### E009 - Smoke-test all 2D methods

**Status:** Completed

**Aim**

- Verify that every supported 2D method can train and test through the unified runner

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cpu
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 7c056d8

**Experiment setup**

- Model: wavecnp2 wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: fake
- Experiment root: smoke_all_2d_methods_20260614
- Datasets: fake
- Seeds: 1
- Results directory: rsults

**Command**

```bash
RUN_ROOT="smoke_all_2d_methods_20260614" DATASETS="fake" SEEDS="1" DEVICE="cpu" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 1 --batch-size 1 --r-dim 16 --level 1 --num-train-images 1 --num-val-images 1 --num-test-images 1
```

**Outputs**

- Output path: _experiments/smoke_all_2d_methods_20260614; aggregate CSV in rsults/

**Results**

- Main metric: All 8 methods produced finite test log-likelihoods
- Secondary metrics:
- Runtime: About 90 seconds
- Peak GPU memory: N/A (CPU)

**Observations**

- Notes: ANP failed once, then completed on retry.

### E008 - Run selected 2D methods on MNIST

**Status:** Failed

**Aim**

- Run selected 2D methods on MNIST for five seeds

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 7c056d8

**Experiment setup**

- Model: wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp
- Task: image
- Dataset / kernel: mnist
- Experiment root: all_2d_methods_20260612_171529
- Datasets: mnist
- Seeds: 1 2 3 4 5
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_2d_methods_20260612_171529" METHODS="wavecnp2new convcnp2 cnp2 anp2 tnp liecnp2 tetnp" DATASETS="mnist" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_2d_all_methods.sh --epochs 10
```

**Outputs**

- Output path: _experiments/all_2d_methods_20260612_171529; aggregate CSV in rsults/

**Results**

- Main metric: 30/35 method/seed runs completed
- Secondary metrics: LieCNP2 failed for all five seeds
- Runtime: About 40 minutes
- Peak GPU memory:

**Observations**

- Notes: The aggregate CSV reports only one retained summary per method because the 2D output paths are reused across seeds.

### E007 - Run all 1D methods on selected GP kernels

**Status:** Stopped

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 8081d89

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260609_200505

- Kernels: matern eq polynomial linear
- Seeds: 1 2 3 4 5
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_1d_gp_kernels_20260609_200505" KERNELS="matern eq polynomial linear" SEEDS="1 2 3 4 5" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 10
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260609_200505; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 60/160 requested runs produced summaries. The aggregate CSV is a progress snapshot, not evidence that the full experiment completed.

### E006 - Run all 1D methods on selected GP kernels

**Status:** Running

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: fa0d4d3

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260608_232620

- Kernels: matern eq polynomial linear
- Seeds: 1 12 123 1234 12345
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
RUN_ROOT="all_1d_gp_kernels_20260608_232620" KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 10
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260608_232620; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 

### E005 - Run all 1D methods on selected GP kernels

**Status:** Running

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: cf82e72

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260607_213023

- Kernels: matern eq polynomial linear
- Seeds: 1 12 123 1234 12345
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260607_213023; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 

### E004 - Run all 1D methods on selected GP kernels

**Status:** Running

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: cf82e72

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260607_144919

- Kernels: matern eq polynomial linear
- Seeds: 1 12 123 1234 12345
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 2
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260607_144919; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- The performance of WaveCNP is good just worse than TNP
- Next, run 20 epochs instead of 2

### E003 - Run all 1D methods on selected GP kernels

**Status:** Running

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: cf82e72

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260607_144911

- Kernels: matern eq polynomial linear
- Seeds: 1 12 123 1234 12345
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260607_144911; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 

### E002 - Run all 1D methods on selected GP kernels

**Status:** Running

**Aim**

- Run all 1D methods on selected GP kernels

**Server and environment**

- Server alias: jarvis1.ihpc.uts.edu.au
- Hostname: jarvis1.ihpc.uts.edu.au
- Device(s): cuda:0
- Workspace path: /data/juxuan/workspace_py/wavecnp-git
- Git commit: 1594617

**Experiment setup**

- Model: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Task: regression
- Dataset / kernel: matern eq polynomial linear
- Experiment root: all_1d_gp_kernels_20260607_122254

- Kernels: matern eq polynomial linear
- Seeds: 1 12 123 1234 12345
- Methods: convcnp cnp anp tnp liecnp tetnp wavecnp wavecnpnew
- Results directory: rsults

**Command**

```bash
KERNELS="matern eq polynomial linear" SEEDS="1 12 123 1234 12345" DEVICE="cuda:0" RESULTS_DIR="rsults" bash scripts/run_1d_all_methods.sh --epochs 20
```

**Outputs**

- Output path: _experiments/all_1d_gp_kernels_20260607_122254; aggregate CSV in rsults/

**Results**

- Main metric:
- Secondary metrics:
- Runtime:
- Peak GPU memory:

**Observations**

- Notes: 

### E001 - Short descriptive title

**Status:** Planned / Running / Completed / Failed / Cancelled

**Aim**

- What question does this experiment answer?
- What result would count as success?
- What baseline or previous experiment is it compared against?

**Server and environment**

- Server alias:
- Hostname:
- Device(s):
- GPU memory:
- Workspace path:
- Git commit:
- Python environment:
- Important package versions:

**Experiment setup**

- Model:
- Task:
- Dataset / kernel:
- Config file:
- Changed hyperparameters:
- Seed(s):
- Expected runtime:
- Expected disk usage:

**Command**

```bash
# Run from the project root unless otherwise noted.
python train.py --task regression --data matern --model wavecnp --epochs 200 --device cuda:0
```

**Outputs**

- Log file:
- Checkpoint directory:
- Result CSV / JSON:
- Plot directory:
- TensorBoard / wandb link:

**Results**

- Main metric:
- Secondary metrics:
- Best checkpoint:
- Final checkpoint:
- Runtime:
- Peak GPU memory:

**Observations**

- What happened during training?
- Any instability, warnings, crashes, or unusual curves?
- What should be tried next?

## Result Files

Keep stable artifact paths here so they are easy to find later.

- `rsults/1d_all_methods_gp_kernels_smoke_format_20260606_134446.csv` - aggregated 1D results for GP kernel experiments
- `rsults/1d_all_methods_matern_all_1d_matern_20260605_133111.csv` - aggregated Matern kernel results

## Naming Conventions

- Use experiment IDs like `E001`, `E002`, `E003`.
- Use server aliases consistently, for example `lab-a100-1`, `lab-3090-2`, or `cloud-l4-1`.
- Include the experiment ID in output folders when possible, for example `results/E001_wavecnp_matern_seed0/`.
- Record commands exactly as run, including environment variables and changed flags.

## Automatic Logging

`train.py` and `scripts/run_1d_all_methods.sh` add a row to `Running
Experiments` and a detailed entry to `Experiment Log` when a real run starts.
The all-method runner moves its row to `Completed Experiments` only after every
requested combination succeeds. Failed runner exits move to `Incomplete
Experiments`. Dry runs are not logged.

- Set `SERVER_ALIAS=my-server` to control the server name written to this file.
- Set `EXPERIMENT_AIM="short aim"` to override the default aim.
- Set `EXPERIMENTS_LOG=0` to skip logging for a scratch run.
- For `scripts/run_1d_all_methods.sh`, one batch-level record is added instead
  of one record for every model, kernel, and seed.

## Workflow

1. Add the server to `Server Inventory` if it is new.
2. Add a row to `Running Experiments` before launching the command.
3. Paste a full entry into `Experiment Log`.
4. When the run succeeds, move it to `Completed Experiments`; otherwise move it
   to `Incomplete Experiments`.
5. Fill in metrics, artifact paths, observations, and next steps.
