# open-audio-sr

**AudioSR**([haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution)、
コードMITライセンス・`basic`学習済み重み[haoheliu/audiosr_basic](https://huggingface.co/haoheliu/audiosr_basic)は
Apache-2.0)という実在する音声帯域拡張(オーディオ超解像)AIモデルを、Rust + RPoemでラップして
他プロジェクトから使えるようにする。

## これは何か / 何ではないか(正直な開示)

- **これは実在するAIモデルの呼び出し**であり、AudioSR自体の再学習・改造はしていない。
- AudioSRは**拡散モデル**(単純な線形補間・EQブーストではなく、実際に高域の情報を推定・生成する)。
  2026-09-26に実機検証済み: 8kHzに帯域制限した実音声(パブリックドメイン録音)を入力し、
  単純アップサンプル(sinc補間、対照群)では8kHz以上が-59.3dB(実質無音)のままなのに対し、
  AudioSR適用後は-24.8dB(元の全帯域録音の-21.0dBに近い水準)——**単なる補間ではなく、
  AIが実際に高域のエネルギーを生成していることを定量的に確認済み**。
- **AudioSRの実際の限界は約24kHz帯域(48kHzサンプルレート)まで。** これを超えるサンプルレート
  (例: 192kHz)を要求しても、AIが生成した本物の高域情報が増えるわけではない(ユーザー確認済み、
  2026-09-26)。
- **本体はPythonのAudioSR実装(拡散モデル)をサブプロセスとして呼んでいるだけ**で、Rustでの
  ネイティブ推論(ONNX化・candle移植等)はしていない。`open-cuda`/`open-directx`によるGPU推論・
  `open-cpu`(`Cargo.toml`に依存として入れてあるが未使用)によるAVX2/AVX512判定を活かした
  高速化は、今後この委譲を置き換える作業として残っている。
- **`aruaru-llm`(言語モデル)はここでは一切使わない**——音声波形を生成できないため。
- この開発機のGPUは古く(検出したドライバはCUDA 11.4世代)、現行PyTorchが要求する計算能力を
  満たさない可能性が高いため、既定はCPU推論。**CPU推論は実測で、10 DDIMステップ・約5秒の
  音声1件あたり数十秒〜数分**(モデル読み込みを含めると初回はもっとかかる)。

## 使い方

### 1. Python側のセットアップ(初回のみ)

```bash
"C:\Users\<user>\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install --no-deps audiosr
# audiosr自身のrequirements(numpy<=1.23.5等)がPython 3.13と非互換なため、
# 実際に必要なものだけを現代的なバージョンで個別に入れる:
.venv\Scripts\python.exe -m pip install numpy torchaudio tqdm pyyaml einops chardet soundfile ^
  progressbar2 librosa huggingface_hub unidecode pandas phonemizer progressbar timm torchlibrosa ^
  transformers ftfy typer tokenizers wcwidth matplotlib torchcodec pyparsing cycler kiwisolver ^
  contourpy fonttools
```

初回実行時にモデル重み(`basic`、約5.8GB。CLAP/RoBERTa等のテキストエンコーダ込み)を
Hugging Faceから自動ダウンロードする。

### 2. ライブラリとして(同一プロセス内)

```rust
use open_audio_sr::{super_resolve_file, SrOptions};
use std::path::Path;

super_resolve_file(
    Path::new("input.wav"),
    Path::new("output.wav"),
    &SrOptions::default(),
)?;
```

### 3. HTTPサービスとして(他プロセス・他言語から)

```bash
bash scripts/fetch-deps.sh   # RPoem等の依存を .deps/ へ固定コミットで取得(初回のみ)
cargo run --release          # 127.0.0.1:4620 で待受
curl -X POST --data-binary @input.wav "http://127.0.0.1:4620/v1/super_resolve?guidance_scale=3.5&ddim_steps=10" -o output.wav
```

| API | 内容 |
|---|---|
| `POST /v1/super_resolve?model=basic\&guidance_scale=3.5\&ddim_steps=10\&seed=42` | bodyに音声ファイルのバイト列。帯域拡張後のWAV(48kHz)を返す |
| `GET /v1/health` | Python venv・推論スクリプトが揃っているか(`open_audio_sr::is_available()`) |
| `GET /healthz` | 死活確認 |

## 依存関係(deps.lock)

`RPoem`(HTTP API用、`open-runo-poem-compat`経由)・`open-web-server`/`open-raid-z`/`RS-SmartTCP`
(RPoemの推移的依存)・`open-cpu`(将来のCPU命令セット活用の下地)を`aruaru-search`と同じ
コミットに固定して`.deps/`へ取得する(`scripts/fetch-deps.sh`)。

## 既知の課題 / 次にやること

1. **GUIの実クリック操作・HTTPサービスの実機検証がまだ**(ライブラリ関数`super_resolve_file`単体
   での実推論・音質改善は2026-09-26に実機検証済みだが、RPoem経由のHTTP APIそのものは未検証)。
2. GPU(CUDA)推論への切り替え(`open-cuda`)、DirectX経由の推論(`open-directx`)、CPU命令セット
   判定(`open-cpu`)を活かした高速化はいずれも未着手。
3. AudioSRの`speech`モデル(音声特化重み)は未検証(`basic`のみ検証済み)。
4. Windows Developer Mode未有効時、Hugging Face Hubのキャッシュがsymlinksを使えず警告が出る
   (動作はする、ディスク容量が余分にかかるだけ)。
