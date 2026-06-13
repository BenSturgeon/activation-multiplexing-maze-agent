"""
Modal script to generate the disinhibition GIF.

Usage:
    modal run scripts/modal_gif.py
"""

import modal

app = modal.App("disinhibition-gif")

LOCAL_DIR = "/Users/bensturgeon/werk/activation-multiplexing-maze-agent"

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
        "Pillow>=9.0.1",
        "imageio",
        "tqdm",
        "scikit-learn>=0.23.2",
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
        "/Users/bensturgeon/werk/rl-mech-interp/full_run",
        remote_path="/root/project/all_checkpoints",
    )
)


@app.function(
    image=image,
    gpu="L4",
    timeout=1800,
)
def generate_gif(seed: int = 42, channel: int = 18, max_steps: int = 200, fps: int = 8):
    """Run the disinhibition GIF generation script."""
    import subprocess
    import os
    os.chdir("/root/project")

    cmd = (
        f"python scripts/generate_disinhibition_static.py "
        f"--model_path all_checkpoints/model_60001.pt "
        f"--channel {channel} "
        f"--seed 107 "
        f"--output /root/project/disinhibition_output.png"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {result.returncode}")

    # Return both PNG and GIF
    files = {}
    for ext in ['png', 'gif']:
        path = f"/root/project/disinhibition_output.{ext}"
        if os.path.exists(path):
            with open(path, "rb") as f:
                files[ext] = f.read()
            print(f"  {ext}: {os.path.getsize(path)} bytes")
    return files


@app.local_entrypoint()
def main(seed: int = 42, channel: int = 18, max_steps: int = 200, fps: int = 8):
    """Entry point."""
    print(f"Generating disinhibition GIF (seed={seed}, channel={channel})...")
    gif_bytes = generate_gif.remote(seed=seed, channel=channel, max_steps=max_steps, fps=fps)

    base = "/Users/bensturgeon/Documents/obsidian_vault/files/thesis_figures/disinhibition"
    for ext, data in gif_bytes.items():
        path = f"{base}_60k.{ext}"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved {ext} to {path} ({len(data)} bytes)")
