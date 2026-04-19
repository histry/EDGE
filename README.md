Feature 1: 移除了 4800 维的重型特征，实现了 Wav2Vec2 + Librosa 的 803 维轻量级混合特征管线。

Feature 2: 底层植入 sMDM 架构与 Lipschitz 谱归一化，支持 63 关节 (381维) 敦煌舞关键帧丝滑插值。

Feature 3: 彻底解决显存瓶颈，完美适配单卡 RTX 4090 炼丹。