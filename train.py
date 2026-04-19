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
        train_stage=opt.train_stage,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)