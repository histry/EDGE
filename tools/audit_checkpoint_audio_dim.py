import argparse
import json
import os
import sys

import torch


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("ema_state_dict", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint, "plain_state_dict"
    raise ValueError(
        "Unsupported checkpoint format. Expected plain state_dict or dict with "
        "ema_state_dict/model_state_dict/state_dict."
    )


def normalize_key(key):
    return key[len("module."):] if key.startswith("module.") else key


def find_key(state_dict, target_key):
    for key in state_dict.keys():
        if normalize_key(key) == target_key:
            return key
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Audit whether checkpoint audio projection matches requested audio_dim."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--out_json", default="")
    parser.add_argument(
        "--fail_on_mismatch",
        action="store_true",
        help="Exit with non-zero status when audio projection is missing or mismatched.",
    )
    args = parser.parse_args()

    checkpoint = torch_load(args.checkpoint)
    state_dict, state_source = extract_state_dict(checkpoint)

    cond_weight_key = find_key(state_dict, "cond_projection.weight")
    cond_bias_key = find_key(state_dict, "cond_projection.bias")

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "state_source": state_source,
        "requested_audio_dim": int(args.audio_dim),
        "cond_projection_weight_key": cond_weight_key,
        "cond_projection_bias_key": cond_bias_key,
        "status": "unknown",
        "message": "",
    }

    if cond_weight_key is None:
        report["status"] = "missing"
        report["message"] = (
            "cond_projection.weight was not found in the checkpoint. "
            "The audio projection layer will be newly initialized."
        )
    else:
        weight = state_dict[cond_weight_key]
        if not torch.is_tensor(weight) or weight.ndim != 2:
            report["status"] = "invalid_shape"
            report["checkpoint_cond_projection_shape"] = str(getattr(weight, "shape", None))
            report["message"] = "cond_projection.weight exists but is not a 2D tensor."
        else:
            out_dim, in_dim = int(weight.shape[0]), int(weight.shape[1])
            report["checkpoint_cond_projection_out_dim"] = out_dim
            report["checkpoint_cond_projection_audio_dim"] = in_dim

            if in_dim == args.audio_dim:
                report["status"] = "compatible"
                report["message"] = (
                    f"Audio projection is compatible: checkpoint audio_dim={in_dim}, "
                    f"requested audio_dim={args.audio_dim}."
                )
            else:
                report["status"] = "mismatch"
                report["message"] = (
                    f"Audio projection mismatch: checkpoint audio_dim={in_dim}, "
                    f"requested audio_dim={args.audio_dim}. "
                    "cond_projection.weight will be shape-mismatched and reinitialized "
                    "by the compatibility loader. Do not claim full inherited music encoding ability."
                )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    if args.fail_on_mismatch and report["status"] != "compatible":
        sys.exit(2)


if __name__ == "__main__":
    main()