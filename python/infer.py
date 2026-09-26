#!/usr/bin/env python
"""open-audio-sr: AudioSR(haoheliu/versatile_audio_super_resolution)の推論を
コマンドラインから呼べるようにした薄いラッパー。

正直な開示:
- ここでやっているのは実在するAudioSR(拡散モデル、コードMIT・basic重みApache-2.0)の
  呼び出しだけで、モデル自体の再学習・改造はしていない。
- torchaudio(2.x)の既定の音声読み込みバックエンドはtorchcodecだが、torchcodecは
  FFmpegの共有ライブラリ(DLL)を必要とし、この開発機には入っていない
  (`ffmpeg.exe`という実行ファイルはあるが、torchcodecが必要とするのは
  `avcodec-*.dll`等の共有ライブラリでこれとは別物)。soundfileは共有ライブラリ無しで
  素朴なWAV読み込みができるため、`torchaudio.load`をsoundfile版へ差し替えて回避する
  (monkeypatch、AudioSRのコード自体は変更しない)。
"""
import argparse
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _load_via_soundfile(path, *_args, **_kwargs):
    """torchaudio.loadの代わり。(channels, samples)のfloat32テンソルとサンプルレートを返す
    (torchaudio.loadと同じ形)。torchcodec(FFmpeg共有ライブラリが要る)を経由しない。"""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfileは(samples, channels)で返すので転置してtorchaudio互換の(channels, samples)にする
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, sr


torchaudio.load = _load_via_soundfile

import audiosr  # noqa: E402 (monkeypatch後にimportする必要がある)


def main() -> int:
    p = argparse.ArgumentParser(description="AudioSRで音声を帯域拡張する")
    p.add_argument("input", help="入力音声ファイル(WAV等)")
    p.add_argument("output", help="出力先(48kHz WAV)")
    p.add_argument("--model", default="basic", choices=["basic", "speech"], help="AudioSR公式の重み名")
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--ddim-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="cpu または cuda(このリポジトリの既定はcpu、README参照)")
    args = p.parse_args()

    print(f"loading model ({args.model}) on {args.device}...", flush=True)
    sr_handle = audiosr.build_model(model_name=args.model, device=args.device)
    print("model loaded, running inference...", flush=True)
    waveform = audiosr.super_resolution(
        sr_handle,
        args.input,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
    )
    print("inference done, shape:", waveform.shape, flush=True)
    out = waveform[0].T if waveform.ndim == 3 else waveform.T
    sf.write(args.output, out, 48000)
    print("wrote", args.output, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
