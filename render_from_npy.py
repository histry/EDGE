import os
import argparse
import numpy as np
import torch
from vis import skeleton_render, SMPLSkeleton
from dataset.quaternion import ax_from_6v


def _as_batch_motion(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[-1] != 151:
            raise ValueError(f"Expected [T,151], got {arr.shape}")
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[-1] != 151:
            raise ValueError(f"Expected [B,T,151], got {arr.shape}")
        return arr
    raise ValueError(f"Expected [T,151] or [B,T,151], got {arr.shape}")


def _sanitize_contacts(contacts, seq_len):
    contacts = np.asarray(contacts)
    if contacts.ndim == 1:
        contacts = np.repeat(contacts[:, None], 4, axis=1)
    if contacts.ndim == 2 and contacts.shape[1] == 1:
        contacts = np.repeat(contacts, 4, axis=1)
    if contacts.ndim != 2:
        return np.zeros((seq_len, 4), dtype=np.float32)
    if contacts.shape[1] != 4:
        if contacts.shape[1] > 4:
            contacts = contacts[:, :4]
        else:
            pad = np.zeros(
                (contacts.shape[0], 4 - contacts.shape[1]),
                dtype=contacts.dtype,
            )
            contacts = np.concatenate([contacts, pad], axis=1)
    if contacts.shape[0] == seq_len - 1:
        contacts = np.concatenate([contacts, contacts[-1:]], axis=0)
    elif contacts.shape[0] < seq_len:
        pad_src = (
            contacts[-1:]
            if len(contacts)
            else np.zeros((1, 4), dtype=np.float32)
        )
        contacts = np.concatenate(
            [
                contacts,
                np.repeat(pad_src, seq_len - contacts.shape[0], axis=0),
            ],
            axis=0,
        )
    elif contacts.shape[0] > seq_len:
        contacts = contacts[:seq_len]
    return contacts.astype(np.float32)


def _render_one(
    motion_data,
    audio,
    output,
    camera_mode,
    render_smooth_window,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    motion_tensor = torch.tensor(
        motion_data, dtype=torch.float32, device=device
    ).unsqueeze(0)
    contacts = _sanitize_contacts(
        motion_tensor[0, :, 0:4].detach().cpu().numpy(),
        motion_data.shape[0],
    )
    pos = motion_tensor[:, :, 4:7]
    seq_len = pos.shape[1]
    q_6d = motion_tensor[:, :, 7:].reshape(1, seq_len, 24, 6)
    q_ax = ax_from_6v(q_6d)
    smpl = SMPLSkeleton(device=device)
    poses_3d = smpl.forward(q_ax, pos).detach().cpu().numpy()[0]
    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(output).replace(".mp4", "")
    skeleton_render(
        poses=poses_3d,
        epoch=base_name,
        out=out_dir,
        name=[audio],
        sound=True,
        stitch=False,
        contact=contacts,
        render=True,
        camera_mode=camera_mode,
        output_path=output,
        render_smooth_window=max(1, int(render_smooth_window)),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Render EDGE 151D motion. Use --render_smooth_window 1 for "
            "scientific inspection; 3-5 only for presentation rendering."
        )
    )
    parser.add_argument("--motion", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--camera_mode",
        type=str,
        choices=["fixed", "follow"],
        default="fixed",
    )
    parser.add_argument(
        "--render_smooth_window",
        type=int,
        default=1,
        help="1 disables render-only smoothing; use 3 or 5 for display copies.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.motion):
        raise FileNotFoundError(f"Motion not found: {args.motion}")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    print(f"🎬 正在读取动作张量: {args.motion}")
    motion_batch = _as_batch_motion(np.load(args.motion, allow_pickle=True))
    print(f"motion shape: {motion_batch.shape}")
    if motion_batch.shape[0] == 1:
        print(
            f"🎥 开始渲染 (camera={args.camera_mode}, "
            f"smooth={args.render_smooth_window})..."
        )
        _render_one(
            motion_batch[0],
            args.audio,
            args.output,
            args.camera_mode,
            args.render_smooth_window,
        )
        print(f"✅ 视频渲染完成: {args.output}")
    else:
        root, ext = os.path.splitext(args.output)
        for i in range(motion_batch.shape[0]):
            out = f"{root}_b{i:02d}{ext or '.mp4'}"
            _render_one(
                motion_batch[i],
                args.audio,
                out,
                args.camera_mode,
                args.render_smooth_window,
            )
        print(f"✅ batch 视频渲染完成，共 {motion_batch.shape[0]} 条。")


if __name__ == "__main__":
    main()
