//! open-audio-sr: AudioSR(haoheliu/versatile_audio_super_resolution、コードMIT・重みApache-2.0の
//! 実在する帯域拡張AIモデル)をRustから呼び出すためのラッパー。
//!
//! **正直な開示 / Honest disclosure**:
//! - 推論本体はPythonのAudioSR実装(拡散モデル)をサブプロセスとして呼び出しているだけで、
//!   Rustでの再実装(ネイティブ推論)はしていない。`open-cuda`/`open-directx`によるGPU推論・
//!   `open-cpu`によるAVX2/AVX512判定を活かした高速化は、このPythonサブプロセス委譲を置き換える
//!   将来の作業であり、現時点では未着手(`Cargo.toml`の`open-cpu`依存は今後のための下地のみ)。
//! - `aruaru-llm`(言語モデル)は音声波形を生成できないため、ここでは一切使わない。
//! - AudioSRの実際の限界は約24kHz帯域(48kHzサンプルレート)まで。それを超えるサンプルレートを
//!   要求しても、AIが生成した本物の高域情報は増えない(単純な帯域制限/無音になるだけ)。
//! - GPU(CUDA)は本開発機のGPUが古く(検出したドライバはCUDA 11.4世代)、現行PyTorchの多くの
//!   ビルドが要求する計算能力を満たさない可能性が高いため、既定はCPU推論(遅い)。
//!
//! ライブラリとして直接呼ぶ(同一プロセス内、プロセス起動のオーバーヘッドはあるがネットワーク不要)
//! ことも、`open-audio-sr`をHTTPサービスとして立ち上げて他プロセス・他言語から呼ぶことも、
//! どちらもできるように設計している(`super_resolve_file`が両方から使われる共通の実体)。

use std::path::{Path, PathBuf};
use std::process::Command;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SrError {
    #[error("Python(venv)を実行できません: {0}")]
    Spawn(String),
    #[error("AudioSRの推論に失敗しました: {0}")]
    Inference(String),
    #[error("出力ファイルが見つかりません: {0}")]
    MissingOutput(String),
}

#[derive(Debug, Clone)]
pub struct SrOptions {
    /// 拡散モデルの分類器フリーガイダンス強度(既定3.5、AudioSR公式デモの既定値)。
    pub guidance_scale: f32,
    /// DDIMのサンプリングステップ数(既定10、少ないほど速いが粗くなる)。
    pub ddim_steps: u32,
    pub seed: i64,
    /// `basic`(既定)または`speech`(音声特化、AudioSR公式が配布する重み名)。
    pub model_name: String,
}

impl Default for SrOptions {
    fn default() -> Self {
        SrOptions { guidance_scale: 3.5, ddim_steps: 10, seed: 42, model_name: "basic".to_string() }
    }
}

/// このリポジトリのPython venv(`.venv/`)と推論スクリプト(`python/infer.py`)の場所。
/// リポジトリ直下からの相対で決め打ちにせず、`OPEN_AUDIO_SR_HOME`環境変数で上書きできるようにする
/// (他プロジェクトのCargo.tomlからpath依存で組み込んだときに、実行時カレントディレクトリが
/// このリポジトリの外になっても動くように)。
fn repo_home() -> PathBuf {
    if let Ok(dir) = std::env::var("OPEN_AUDIO_SR_HOME") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn venv_python() -> PathBuf {
    let home = repo_home();
    #[cfg(windows)]
    {
        home.join(".venv").join("Scripts").join("python.exe")
    }
    #[cfg(not(windows))]
    {
        home.join(".venv").join("bin").join("python")
    }
}

/// `input`(WAV等、ffmpeg/torchaudioが読める形式)をAudioSRで帯域拡張し、`output`(WAV、48kHz)へ書く。
/// 同期・ブロッキング呼び出し(Pythonサブプロセスの終了を待つ)。CPU推論だと数分〜数十分かかりうる
/// (このクレートの`README.md`の既知の制限を参照)。
pub fn super_resolve_file(input: &Path, output: &Path, opts: &SrOptions) -> Result<(), SrError> {
    let python = venv_python();
    let script = repo_home().join("python").join("infer.py");
    let status = Command::new(&python)
        .arg(&script)
        .arg(input)
        .arg(output)
        .arg("--model")
        .arg(&opts.model_name)
        .arg("--guidance-scale")
        .arg(opts.guidance_scale.to_string())
        .arg("--ddim-steps")
        .arg(opts.ddim_steps.to_string())
        .arg("--seed")
        .arg(opts.seed.to_string())
        .status()
        .map_err(|e| SrError::Spawn(format!("{} ({})", e, python.display())))?;
    if !status.success() {
        return Err(SrError::Inference(format!("終了コード{:?}", status.code())));
    }
    if !output.exists() {
        return Err(SrError::MissingOutput(output.display().to_string()));
    }
    Ok(())
}

/// このマシンで実際に推論可能か(Python venvと推論スクリプトが揃っているか)の軽い確認。
/// 実際にモデルを読み込むわけではないので高速(HTTPの`/healthz`・呼び出し前の事前チェック用)。
pub fn is_available() -> bool {
    venv_python().exists() && repo_home().join("python").join("infer.py").exists()
}
