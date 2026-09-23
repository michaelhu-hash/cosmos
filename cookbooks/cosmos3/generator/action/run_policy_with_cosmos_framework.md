# Cosmos3-Policy-DROID server

Cosmos 3 offers two post-trained policy models for DROID:
1. [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
2. [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)

Each of these policy models is served by a policy **Server** that streams actions to a **Client** driving a simulated or real robot. This example uses [`RoboLab`](https://github.com/NVlabs/RoboLab), a simulation benchmark for task-generalist policies, as the client. Start the server first, then connect the client.

## Table of Contents

- [Policy Server](#server)
- [Simulation Client](#client)

## Policy Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```bash
docker build \
  --build-arg INSTALL_APEX=0 \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```bash
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-torch213-train \
      --group=policy-server && \
    exec bash; \
  '
```

Inside the container, start the policy server:

1. For [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --port 8000
   ```

2. For [Cosmos3-Edge-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID), run:

   ```
   python -m cosmos_framework.scripts.action_policy_server_robolab \
     --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
     --port 8000 \
     --format-prompt-as-json True \
     --guidance-interval 960 1001
   ```

   The guidance interval applies classifier-free guidance only to denoising
   timesteps in the inclusive range `[960, 1001]`. Omit
   `--guidance-interval` to apply guidance at every denoising step.

## Simulation Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```bash
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```bash
./docker/build_docker.sh latest
```

Launch the container:

```bash
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

To evaluate across multiple sub-environments in parallel in headless mode:

```bash
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```

Example output:

<video controls width="864" height="480" src="https://github.com/user-attachments/assets/95a16737-5eb9-4b3f-a0ad-3a6b929b423f"></video>

## Cosmos3-Nano-HumanAction (human hands and body)

[Cosmos3-Nano-HumanAction](https://huggingface.co/nvidia/Cosmos3-Nano-HumanAction) runs policy through the same
`cosmos_framework.scripts.inference` entrypoint as forward and inverse dynamics, with `model_mode: "wam"` (video and
actions are sampled jointly from one conditioning frame plus the task caption). The three domains and their readers are
described in the forward-dynamics notebook; the spec below uses the Action100M hand example. `raw_action_dim` is 48 for
`webhumanaction_hand`, 57 for `webhumanaction_body` and `hand_pose`.

```jsonl
{"name": "humanaction_hand_policy", "model_mode": "wam", "prompt": "<task or subtask caption>", "vision_path": "/abs/path/first_frame.png", "domain_name": "webhumanaction_hand", "raw_action_dim": 48, "view_point": "ego_view", "fps": 15, "action_chunk_size": 72, "image_size": 480, "num_frames": 73, "resolution": "480", "seed": 0, "num_steps": 30, "guidance": 1.0, "shift": 5.0}
```

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency --no-guardrails \
  -i humanaction_policy.jsonl -o outputs/humanaction_policy \
  --checkpoint-path nvidia/Cosmos3-Nano-HumanAction \
  --seed 0
```

The rollout video is written to `outputs/humanaction_policy/humanaction_hand_policy/0/vision.mp4` and the normalized
action chunk to `sample_outputs.json` (`outputs[0].content.action`, `[72, 48]`). Invert the normalization with the
matching reader (`WebHumanActionHandLeRobotDataset(...).denormalize(actions)`).
