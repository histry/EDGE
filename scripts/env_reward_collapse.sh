export RUN_DIR=runs/train_stage45/stage45_riskfix_v1_e504
export CKPT_MAIN=$RUN_DIR/weights/train-10.pt

export DATA_DIR=data/dunhuang_bvh/processed
export START_POSE=test_keyframes/demo_dyl002_start.npy
export END_POSE=test_keyframes/demo_dyl002_end.npy

export MUSIC2=test_music_bank/dunhuangwu2.wav
export MUSIC3=test_music_bank/dunhuangwu3.wav
export MUSIC4=test_music_bank/dunhuangwu4.wav

export MUSIC2_FEAT=test_music_bank/dunhuangwu2.npy
export MUSIC3_FEAT=test_music_bank/dunhuangwu3.npy
export MUSIC4_FEAT=test_music_bank/dunhuangwu4.npy

export TRAJ="0,0;0.5,0.7;-0.3,1.2;0,1.6"

export RAG_EXPR=data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr.npz
export OUT_ROOT=output/reward_collapse

mkdir -p "$OUT_ROOT"
