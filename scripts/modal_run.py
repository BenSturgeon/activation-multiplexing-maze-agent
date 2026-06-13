"""
Modal stub for running GPU experiments remotely.

Usage:
    modal run scripts/modal_run.py --command "python scripts/smoke_test.py"
    modal run scripts/modal_run.py --command "python sweep_offset_experiment.py --offset_start -2 --offset_end 2 --offset_step 1"
"""

import modal
import subprocess

app = modal.App("maze-agent")

# Local path to this repo on the machine running `modal run` (add_local_dir runs locally).
PROJECT_DIR = "/Users/bensturgeon/werk/activation-multiplexing-maze-agent"

# Persistent volume for generated artifacts (figures/CSVs). Without this, anything
# written inside the container is lost on exit. Write outputs to /root/project/outputs
# in your scripts, then pull with: modal volume get rl-mech-interp-results <path> .
results_vol = modal.Volume.from_name("rl-mech-interp-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "cmake",
        "qtbase5-dev", "qttools5-dev", "qttools5-dev-tools",
        "qtbase5-dev-tools", "libqt5opengl5-dev",
        "build-essential", "g++",
        "libosmesa6-dev", "libgl1-mesa-glx",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "numpy<2.0",
        "scipy>=1.8.0",
        "pandas>=1.3.5",
        "procgen==0.10.7",
        "stable-baselines3==2.2.1",
        "shimmy==1.3.0",
        "gym",
        "gym3",
        "einops",
        "circrl==1.0.0",
        "scikit-learn>=0.23.2",
        "wandb>=0.16.3",
        "matplotlib>=3.5.1",
        "seaborn>=0.13.2",
        "opencv-python>=4.9.0.80",
        "Pillow>=9.0.1",
        "imageio",
        "tqdm",
        "rich",
        "PyYAML",
    )
    .env({
        "MUJOCO_GL": "osmesa",
        "MESA_GL_VERSION_OVERRIDE": "3.3",
        "PYTHONPATH": "/root/project/src:/root/project",
    })
    .add_local_dir(
        PROJECT_DIR,
        remote_path="/root/project",
        ignore=[
            ".venv/", "__pycache__/", ".git/", "slurm_logs/", "wandb/",
        ],
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=14400,
    volumes={"/root/project/outputs": results_vol},
)
def run_command(command: str):
    """Run a shell command in the project directory. Artifacts written under
    /root/project/outputs persist to the rl-mech-interp-results volume."""
    import os
    os.makedirs("/root/project/outputs", exist_ok=True)
    os.chdir("/root/project")

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    results_vol.commit()  # persist any outputs before returning
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result.stdout


@app.local_entrypoint()
def main(command: str, gpu: str = "L4"):
    """Entry point for modal run."""
    print(f"Running on Modal with GPU={gpu}: {command}")
    result = run_command.remote(command)
    print(result)
