import os
import tarfile
import requests
from tqdm import tqdm

# 1. 设置目标文件夹
output_dir = "data/gtzan_pure_wav"
os.makedirs(output_dir, exist_ok=True)

url = "https://hf-mirror.com/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
tar_path = "genres.tar.gz"

# 2. 下载数据包
if not os.path.exists(tar_path):
    print("正在建立连接，准备下载全量数据包...")
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(tar_path, "wb") as file, tqdm(
        desc="下载 GTZAN 数据包", total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            bar.update(file.write(data))
else:
    print("✅ 检测到本地已存在压缩包，跳过下载...")

print("\n🔍 开启严格字节级质检，过滤所有残次品...")

valid_count = 0
discard_count = 0

# 3. 边解压边质检
with tarfile.open(tar_path, "r:gz") as tar:
    # 粗筛：只看不是 ._ 开头且后缀是 .wav 的文件
    potential_files = [
        m for m in tar.getmembers() 
        if m.isfile() and not os.path.basename(m.name).startswith("._") and m.name.endswith(".wav")
    ]
    
    for member in tqdm(potential_files, desc="提取进度"):
        f = tar.extractfile(member)
        if f is not None:
            # 读取文件头部的 12 个字节进行“血统认证”
            header = f.read(12)
            
            # 核心判断：必须同时具备 RIFF 和 WAVE 标识
            if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                # 认证通过，将指针移回文件开头，准备完整保存
                f.seek(0)
                
                # 优化文件名 (例如将 blues.00001.wav 改为 blues_00001.wav)
                filename = os.path.basename(member.name)
                parts = filename.split(".")
                if len(parts) >= 3:
                    clean_name = f"{parts[0]}_{parts[1]}.wav"
                else:
                    clean_name = filename
                    
                dest_path = os.path.join(output_dir, clean_name)
                
                # 直接将二进制流写入本地，速度极快
                with open(dest_path, "wb") as out_f:
                    out_f.write(f.read())
                    
                valid_count += 1
            else:
                # 认证失败（伪装的au、损坏的文件等），直接丢弃
                discard_count += 1

print("\n🎉 质检与提取全部完成！")
print(f"👑 成功提取的纯正 WAV 数量: {valid_count} 首")
print(f"🗑️ 被无情丢弃的残次品数量: {discard_count} 首")
print(f"📂 纯净数据集已就绪: {os.path.abspath(output_dir)}")