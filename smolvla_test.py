#!/usr/bin/env python3
import time
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    print("loaded GB", torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0)
    img_keys = list(getattr(policy.config, "image_features", {}) or [])
    print("image keys", img_keys)
    img = torch.zeros(1, 3, 224, 224, device=device)
    state_dim = getattr(getattr(policy.config, "robot_state_feature", None), "shape", [6])
    if isinstance(state_dim, (list, tuple)):
        state_dim = state_dim[-1]
    state = torch.zeros(1, int(state_dim), device=device)
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok("drive forward\n", return_tensors="pt", padding="max_length", max_length=48)
    batch = {
        "observation.state": state,
        "observation.language.tokens": enc.input_ids.to(device),
        "observation.language.attention_mask": enc.attention_mask.to(
            device=device, dtype=torch.bool
        ),
    }
    for k in img_keys:
        batch[k] = img
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.select_action(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(i, round(dt, 3), out.squeeze().tolist())
    print("slow calls", [round(t, 3) for t in times if t > 0.1])
    if device == "cuda":
        print("peak GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
