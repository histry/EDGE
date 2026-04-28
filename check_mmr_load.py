import argparse
import os
from pathlib import Path

import torch

from EDGE import EDGE
from model.mmr_model import CrossModalMMR


def format_exception(exc):
    return f"{exc.__class__.__name__}: {exc}"


def load_checkpoint(path):
    return torch.load(path, map_location="cpu")


def extract_state_dict(raw_ckpt):
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        return raw_ckpt["model_state_dict"], "wrapped_checkpoint"
    if isinstance(raw_ckpt, dict):
        tensor_like = all(torch.is_tensor(v) for v in raw_ckpt.values()) if raw_ckpt else False
        if tensor_like:
            return raw_ckpt, "plain_state_dict"
    raise ValueError("MMR checkpoint is neither a plain state_dict nor a dict with model_state_dict")


def compare_state_dicts(model_state, expected_state, atol=1e-6, rtol=1e-5):
    if set(model_state.keys()) != set(expected_state.keys()):
        missing = sorted(set(expected_state.keys()) - set(model_state.keys()))
        extra = sorted(set(model_state.keys()) - set(expected_state.keys()))
        return False, {"missing_keys": missing[:10], "extra_keys": extra[:10]}

    mismatched = []
    for key in model_state.keys():
        if not torch.allclose(model_state[key].cpu(), expected_state[key].cpu(), atol=atol, rtol=rtol):
            mismatched.append(key)
            if len(mismatched) >= 10:
                break
    return len(mismatched) == 0, {"mismatched_keys": mismatched}


def check_raw_edge_compatibility(raw_ckpt, device, motion_dim, audio_dim, latent_dim):
    probe_model = CrossModalMMR(
        motion_dim=motion_dim,
        audio_dim=audio_dim,
        latent_dim=latent_dim,
    ).to(device)
    try:
        probe_model.load_state_dict(raw_ckpt, strict=True)
        return True, None
    except Exception as exc:
        return False, format_exception(exc)


def print_header(title):
    print(f"\n========== {title} ==========")


def main():
    parser = argparse.ArgumentParser(description="Verify whether EDGE.py really loads mmr_pretrained.pt")
    parser.add_argument("--mmr_ckpt", type=str, default="weights/mmr_pretrained.pt")
    parser.add_argument("--edge_ckpt", type=str, default="", help="optional main EDGE checkpoint")
    parser.add_argument("--feature_type", type=str, default="hybrid")
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--train_stage", type=str, default="full", choices=["full", "stage1", "stage2"])
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--ema", action="store_true", help="load ema_state_dict for the main EDGE checkpoint if provided")
    opt = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    mmr_path = Path(opt.mmr_ckpt)
    if not mmr_path.is_absolute():
        mmr_path = project_root / mmr_path

    print_header("MMR Load Check")
    print(f"project root : {project_root}")
    print(f"mmr ckpt     : {mmr_path}")
    print(f"main ckpt    : {opt.edge_ckpt or '(none)'}")

    if not mmr_path.exists():
        print("❌ MMR checkpoint file does not exist.")
        return

    raw_ckpt = load_checkpoint(str(mmr_path))

    try:
        normalized_state, ckpt_format = extract_state_dict(raw_ckpt)
        print(f"checkpoint format : {ckpt_format}")
    except Exception as exc:
        print(f"❌ Cannot interpret MMR checkpoint: {format_exception(exc)}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compatible, reason = check_raw_edge_compatibility(
        raw_ckpt=raw_ckpt,
        device=device,
        motion_dim=151,
        audio_dim=opt.audio_dim,
        latent_dim=opt.latent_dim,
    )

    print_header("Raw EDGE Compatibility")
    if compatible:
        print("✅ This file can be loaded by EDGE.py exactly as-is.")
    else:
        print("❌ EDGE.py will fail to load this file as-is.")
        print(f"reason       : {reason}")
        if ckpt_format == "wrapped_checkpoint":
            print("hint         : EDGE.py currently expects a plain state_dict, but train_mmr.py saves a wrapped checkpoint.")

    print_header("Instantiate EDGE")
    try:
        edge = EDGE(
            feature_type=opt.feature_type,
            checkpoint_path=opt.edge_ckpt,
            EMA=opt.ema,
            audio_dim=opt.audio_dim,
            seq_len=opt.seq_len,
            mixed_precision=opt.mixed_precision,
            train_stage=opt.train_stage,
        )
        print("✅ EDGE instantiated successfully.")
    except Exception as exc:
        print("❌ EDGE instantiation failed before/while loading MMR.")
        print(f"reason       : {format_exception(exc)}")
        return

    print_header("Post-load Verification")
    if edge.mmr_model is None:
        print("❌ edge.mmr_model is None. MMR is not active.")
        return

    print("✅ edge.mmr_model is active.")
    print(f"diffusion has mmr model : {edge.diffusion.mmr_model is not None}")
    print(f"same object in diffusion: {edge.diffusion.mmr_model is edge.mmr_model}")
    print(f"eval mode               : {not edge.mmr_model.training}")
    all_frozen = all(not p.requires_grad for p in edge.mmr_model.parameters())
    print(f"all params frozen       : {all_frozen}")

    loaded_ok, details = compare_state_dicts(edge.mmr_model.state_dict(), normalized_state)
    if loaded_ok:
        print("✅ Loaded parameters match the checkpoint content.")
    else:
        print("❌ Loaded parameters do not match the expected checkpoint content.")
        print(f"details      : {details}")

    try:
        dummy_audio = torch.randn(2, opt.seq_len, opt.audio_dim, device=device)
        dummy_motion = torch.randn(2, opt.seq_len, 151, device=device)
        with torch.no_grad():
            audio_latent = edge.mmr_model.encode_audio(dummy_audio)
            motion_latent = edge.mmr_model.encode_motion(dummy_motion)
        print(f"audio latent shape      : {tuple(audio_latent.shape)}")
        print(f"motion latent shape     : {tuple(motion_latent.shape)}")
        print("✅ MMR forward pass works inside EDGE context.")
    except Exception as exc:
        print("❌ MMR forward pass failed after loading.")
        print(f"reason       : {format_exception(exc)}")


if __name__ == "__main__":
    main()
