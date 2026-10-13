# Sigma2 Olivia: first GPU experiment and usage tracking

Step-by-step notes for colleagues • Verified with a successful run on 13 October 2026

## 1. Before starting

You need your own approved NRIS/Sigma2 username, account password, configured authenticator, and access to a project with an Olivia allocation. Do not share accounts, passwords, authenticator QR codes, or recovery codes.

In the commands below, replace `YOUR_USERNAME` with your own username. The example project is `nn12353k` (administrative project name: NN12353K). Use this project only if you are a member; otherwise replace it with your allocated project.

Olivia is a supercomputer with CPU and GPU nodes. The experiment below uses one GPU through the Slurm scheduler.

## 2. Set your account password

Use the account-management links from your Sigma2 registration email or the official password help page:

https://documentation.sigma2.no/getting_help/lost_forgotten_password.html

If no temporary password was generated, follow the account portal's password-setting/reset process. A new password can take about 15 minutes to reach the login nodes.

## 3. Configure two-factor authentication (2FA)

1. Open https://www.metacenter.no and choose **User Login (Passwords)**.
2. Sign in with your own NRIS credentials.
3. Select **Manage OTP secret / 2FA**.
4. Install an authenticator app if needed, and scan the displayed QR code.
5. Enter the current app code to verify setup and submit.
6. Save your recovery codes securely and allow time for synchronization.

Follow the current official instructions if portal labels change:
https://documentation.sigma2.no/getting_help/two_factor_authentication.html

## 4. Connect from Windows PowerShell

Open PowerShell on your computer and enter:

```powershell
ssh YOUR_USERNAME@olivia.sigma2.no
```

On the first connection, compare the host fingerprint against the official SSH documentation before accepting it:
https://documentation.sigma2.no/getting_started/ssh.html

At the **One-time password (OATH)** prompt, enter the current code from your authenticator. At **Password**, enter your NRIS account password. Password/code characters may not appear while typing.

A successful login produces a prompt similar to:

```text
[YOUR_USERNAME@uan02.olivia ~]$
```

You are now on an Olivia login node. All subsequent Bash commands in this guide run here, not in local PowerShell. Do not copy the example prompt itself.

Observed network access: the verified run worked from apartment Wi-Fi without a VPN. This observation does not establish access from every network or from outside Norway; ask NRIS support about your intended location if needed.

## 5. Check your project allocation

Run each command separately, pressing Enter after each:

```bash
hostname
cost
sinfo -o "%P %a %l %D %G"
```

Check that your project appears with available GPU hours. The `sinfo` command lists partitions; GPU jobs use `accel`.

Wait for the terminal prompt to return before entering another command. Do not append `mkdir` to the end of `sinfo` or another existing command.

## 6. Create a working folder

```bash
mkdir -p ~/first_gpu
cd ~/first_gpu
pwd
```

`pwd` should show a path ending in `/first_gpu`. This small test can live in your home folder. Use the documented project/work storage for larger datasets and experiments.

## 7. Create the Python GPU experiment

Paste the entire block, including the final `EOF`, and press Enter:

```bash
cat > gpu_test.py <<'EOF'
import time
import torch

assert torch.cuda.is_available(), "CUDA unavailable: GPU test cannot run."
print("PyTorch:", torch.__version__, flush=True)
print("GPU:", torch.cuda.get_device_name(0), flush=True)

n = 4096
a = torch.randn(n, n, device="cuda")
b = torch.randn(n, n, device="cuda")

for _ in range(3):
    c = a @ b
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    c = a @ b
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print("Result device:", c.device)
print("Result shape:", tuple(c.shape))
print(f"100 GPU matrix multiplications: {elapsed:.3f} seconds")
print(f"Peak PyTorch GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MiB")
print("GPU experiment completed successfully!")
EOF
```

The temporary `>` prompt means Bash is collecting the file contents. It returns to the normal prompt after the final `EOF`. Do not type the displayed `>` characters yourself.

This performs actual matrix multiplications on the GPU. The measured interval covers the 100 multiplications after warm-up, not queue time or the entire job. This is a functional test, not a controlled performance benchmark.

## 8. Create the Slurm GPU job

Use your own project account if different from `nn12353k`:

```bash
cat > gpu_job.sh <<'EOF'
#!/bin/bash -l
#SBATCH --account=nn12353k
#SBATCH --partition=accel
#SBATCH --job-name=first_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=gpu-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONTAINER=/cluster/work/support/container/pytorch_nvidia_25.05_arm64.sif
test -r "$CONTAINER" || { echo "PyTorch container unavailable: $CONTAINER"; exit 1; }

echo "Compute node: $(hostname)"
srun apptainer exec --nv --bind "$PWD:/experiment" \
    "$CONTAINER" python -u /experiment/gpu_test.py
EOF
```

| Setting | Purpose |
|---|---|
| `--account` | Project receiving the usage charge |
| `--partition=accel` | GPU partition |
| `--gpus=1` | Reserve one GPU |
| `--cpus-per-task=4` | Reserve four supporting CPU cores |
| `--mem=16G` | CPU RAM requested; not GPU memory |
| `--time=00:05:00` | Maximum runtime, not mandatory runtime |
| `%j` | Replaced with the job ID in the output filename |

The container path above worked in the verified run. Olivia GPU nodes use ARM architecture, so use software intended for those nodes. If the container becomes unavailable, consult the current NRIS PyTorch guide rather than installing an x86 environment on the login node.

## 9. Submit and monitor

```bash
sbatch gpu_job.sh
```

Example response:

```text
Submitted batch job 2516246
```

Record your actual job ID. Do not resubmit repeatedly to check progress; every submission creates another job.

```bash
squeue -u "$USER"
```

`PD` means pending; `R` means running. A finished job normally disappears from `squeue`. Disappearance alone does not prove success.

Replace `JOB_ID` below with your actual number:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
cat gpu-JOB_ID.out
```

Success: `COMPLETED`, exit code `0:0`, `Result device: cuda:0`, and the successful-completion message.

To cancel an unwanted queued/running job:

```bash
scancel JOB_ID
```

## 10. Verified example

| Item | Observed result |
|---|---|
| Job ID | 2516246 |
| GPU | NVIDIA GH200 120GB |
| PyTorch | 2.8.0a0+5228986c39.nv25.05 |
| Result device | cuda:0 |
| Matrix shape | 4096 × 4096 |
| 100 multiplications | 0.036 seconds |
| Peak PyTorch allocated GPU memory | 288 MiB |
| Queue wait | 12 seconds |
| Entire job runtime | 6 seconds |
| Exit status | 0 |

The `gocryptfs not found` informational message did not prevent this run. Your timings may differ.

## 11. Track shared quota and personal usage

Project overview:

```bash
cost -p nn12353k
```

User-level accounting:

```bash
cost --details
```

Find your username in the detailed output. This is the first place to check personal usage reported by NRIS accounting. Use `cost --man` for output definitions and options. Accounting updates are periodic, and very small usage can round to zero.

For your own job history in this allocation period:

```bash
sacct -u "$USER" -A nn12353k -S 2026-10-01 -X \
  --format=JobID,JobName,Partition,State,Elapsed,AllocTRES%100
```

Change the start date for future allocation periods. `-X` lists allocation records without separately counting `.batch` and application steps.

For a shareable personal job-history export:

```bash
sacct -u "$USER" -A nn12353k -S 2026-10-01 -X -P \
  --format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES \
  > personal_usage.psv
```

This creates a pipe-separated text file; `ElapsedRaw` is seconds. Save periodic copies of your records and retain job logs.

### GPU-hour calculation

Allocated GPU hours = allocated GPU count × elapsed seconds ÷ 3600.

For the verified job: 1 × 6 ÷ 3600 = **0.001667 GPU hours**. The 12-second queue wait is excluded. Use the entire allocation runtime, not the 0.036-second calculation interval.

If `AllocTRES` contains both `gres/gpu:h200=1` and `gres/gpu=1`, they describe the same GPU allocation; count one GPU, not two. GPU hours measure allocated time, not GPU utilization percentage. Failed/cancelled jobs can also consume resources after starting.

### Billing hours versus GPU hours on Olivia

Olivia `accel` jobs use the GPU-hour quota. CPU jobs in `small` and `large` use the billing-hour quota. Billing hours are resource accounting units, not Norwegian kroner, and should not be equated with GPU hours. Use the official accounting output for reported charges; the GPU formula above is a transparent personal estimate.

Suggested record:

| Date | Job ID | Experiment | GPUs | Runtime | Estimated GPU hours | Status |
|---|---|---|---:|---|---:|---|
| 2026-10-13 | 2516246 | Matrix multiplication | 1 | 6 seconds | 0.001667 | Completed |

## 12. Common problems

| Problem | Action |
|---|---|
| Password rejected | Check credentials; allow propagation time after changes |
| OTP rejected | Use a current authenticator code and check phone time synchronization |
| `cd` says no such directory | Run `mkdir -p ~/first_gpu` on its own line, then `cd ~/first_gpu` |
| Output file absent | Check `squeue`; job may not have started, or check the submission directory |
| Job absent from queue | Check `sacct` and the output log |
| Invalid account/partition | Confirm project membership, `cost`, and `sinfo` |
| Container unavailable/CUDA error | Save the log and consult NRIS support; do not treat it as a successful GPU test |

After finishing, `exit` disconnects the SSH session. A submitted batch job can continue independently of your SSH session.

## Official references

- Password help: https://documentation.sigma2.no/getting_help/lost_forgotten_password.html
- 2FA setup: https://documentation.sigma2.no/getting_help/two_factor_authentication.html
- SSH: https://documentation.sigma2.no/getting_started/ssh.html
- GPU job settings: https://documentation.sigma2.no/jobs/job_scripts/olivia/olivia_sample_gpu_job.html
- PyTorch/container guide: https://documentation.sigma2.no/software/application_guides/pytorch_olivia/PyTorchSingleGpu.html
- Usage/accounting: https://documentation.sigma2.no/jobs/projects_accounting.html
- Technical support: support@nris.no

This guide includes no passwords or authenticator secrets. Each colleague must use their own account and authorized project allocation.
