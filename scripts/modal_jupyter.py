"""
Launch a Jupyter server on Modal with GPU and all project dependencies.

Usage:
    modal run scripts/modal_jupyter.py
"""

import modal
import subprocess
import time

app = modal.App("maze-jupyter")

LOCAL_DIR = "/Users/bensturgeon/werk/activation-multiplexing-maze-agent"
MODEL_DIR = "/Users/bensturgeon/werk/rl-mech-interp/full_run"

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
        "procgen==0.10.7",
        "gym",
        "gym3",
        "einops",
        "matplotlib>=3.5.1",
        "seaborn>=0.13.2",
        "scikit-learn>=0.23.2",
        "Pillow>=9.0.1",
        "imageio",
        "tqdm",
        "jupyterlab",
        "ipykernel",
    )
    .env({
        "MUJOCO_GL": "osmesa",
        "MESA_GL_VERSION_OVERRIDE": "3.3",
        "PYTHONPATH": "/root/project/src:/root/project",
    })
    .add_local_dir(
        LOCAL_DIR,
        remote_path="/root/project",
        ignore=[
            ".venv/", "__pycache__/", ".git/", "slurm_logs/", "wandb/",
        ],
    )
    .add_local_dir(
        MODEL_DIR,
        remote_path="/root/project/all_checkpoints",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
)
def run_jupyter():
    """Start a Jupyter server with a tunnel."""
    with modal.forward(8888) as tunnel:
        print(f"\n{'='*60}")
        print(f"Jupyter is running at: {tunnel.url}")
        print(f"{'='*60}\n")

        proc = subprocess.Popen(
            [
                "jupyter", "lab",
                "--no-browser",
                "--port=8888",
                "--ip=0.0.0.0",
                "--allow-root",
                "--NotebookApp.token=''",
                "--NotebookApp.password=''",
                "--notebook-dir=/root/project",
            ],
        )
        proc.wait()


@app.local_entrypoint()
def main():
    run_jupyter.remote()
