from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(
        feature_type=opt.feature_type,
        checkpoint_path=opt.checkpoint,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        audio_dim=opt.audio_dim,
        seq_len=opt.seq_len,
        mixed_precision=opt.mixed_precision,
        gradient_checkpointing=opt.gradient_checkpointing,
        use_sparse_attn=opt.use_sparse_attn,
        sparse_attn_window=opt.sparse_attn_window,
        cond_drop_prob=opt.cond_drop_prob,
        audio_pairing_mode=opt.audio_pairing_mode,
        mmr_loss_weight=opt.mmr_loss_weight,
        keyframe_condition_prob=opt.keyframe_condition_prob,
        keyframe_condition_width=opt.keyframe_condition_width,
        keyframe_loss_weight=opt.keyframe_loss_weight,
        contact_loss_weight=opt.contact_loss_weight,
        foot_loss_weight=opt.foot_loss_weight,
        sync_loss_weight=opt.sync_loss_weight,
        mid_keyframe_condition_prob=opt.mid_keyframe_condition_prob,
        mid_keyframe_count=opt.mid_keyframe_count,
        mid_keyframe_condition_width=opt.mid_keyframe_condition_width,
        mid_keyframe_selection=opt.mid_keyframe_selection,
        beat_guidance_weight=opt.beat_guidance_weight,
        hard_keyframe_project=opt.hard_keyframe_project,
        train_stage=opt.train_stage,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
