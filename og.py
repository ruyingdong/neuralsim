#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

# ———— 设置区 ————
# 改成你自己的路径：
INPUT_ROOT  = Path('/home/david/Grounded-SAM-2/mask')
OUTPUT_ROOT = Path('/media/david/backup/track')

# 支持的类别列表（会跳过不存在的文件夹）
CATEGORIES = ['background', 'frames', 'frames1', 'garment', 'human']
# 每个 chunk 放多少张图
CHUNK_SIZE = 20
# 视频文件夹从 video_0 … video_202
VIDEO_COUNT = 203
# ————————————

def main():
    for idx in range(VIDEO_COUNT):
        vid_name = f'video_{idx}'
        in_dir  = INPUT_ROOT  / vid_name
        out_dir = OUTPUT_ROOT / vid_name

        if not in_dir.is_dir():
            print(f"[跳过] 输入不存在：{in_dir}")
            continue
        if not out_dir.is_dir():
            print(f"[跳过] 输出不存在：{out_dir}")
            continue

        # 1) 读入每个类别里的所有文件（按名字排序）
        files_map = {}
        for cat in CATEGORIES:
            cat_in = in_dir / cat
            if cat_in.is_dir():
                files = sorted([
                    f for f in os.listdir(cat_in)
                    if (cat_in / f).is_file()
                ])
                files_map[cat] = files
            else:
                files_map[cat] = []   # 不存在就空列表

        # 2) 获取输出里的 chunk 文件夹（已按 zero-pad 排序）
        chunks = sorted([
            d for d in os.listdir(out_dir)
            if (out_dir / d).is_dir() and d.startswith('chunk')
        ])

        # 3) 每个 chunk 处理 CHUNK_SIZE 张
        for chunk_idx, chunk_name in enumerate(chunks):
            start = chunk_idx * CHUNK_SIZE
            end   = start + CHUNK_SIZE
            chunk_path = out_dir / chunk_name

            for cat in CATEGORIES:
                cat_chunk = chunk_path / cat
                cat_chunk.mkdir(parents=True, exist_ok=True)

                src_files = files_map.get(cat, [])
                # 复制 [start:end] 范围内的文件
                for fname in src_files[start:end]:
                    src = in_dir  / cat / fname
                    dst = cat_chunk / fname
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"  [错误] {src} → {dst}：{e}")

        print(f"[完成] {vid_name} ⇒ {len(chunks)} 个 chunk，共各类别复制至 {out_dir}")

if __name__ == '__main__':
    main()
