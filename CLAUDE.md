# 開発方針(open-audio-sr)

AudioSR(haoheliu/versatile_audio_super_resolution、コードMIT・basic重みApache-2.0の
実在する拡散モデル)をRust+RPoemでラップする音声超解像サービス。詳細は[README.md](README.md)。

## HANDOFF(2026-09-26 初版)

- **実機検証済み**: 8kHzに帯域制限した実音声(パブリックドメイン録音)→AudioSR適用で
  単純補間(-59.3dB)を大きく上回る-24.8dBの高域エネルギーを実測(元の全帯域-21.0dBに近い)。
  ライブラリ呼び出し(`super_resolve_file`)・HTTP API(`POST /v1/super_resolve`、RPoem経由)
  ともエンドツーエンドで実際に動作することを確認済み(HTTP 200・正しい48kHz WAV)。
- **Python環境構築の要点**: 64bit Python 3.13必須(32bitは不可、torch非対応)。`audiosr`
  パッケージ自身のrequirements(numpy<=1.23.5・transformers==4.30.2等)がPython 3.13と
  非互換のため`--no-deps`で入れ、実際に必要なものだけ現代的なバージョンで個別解決
  (`python/requirements.txt`参照)。`torchaudio.load`の既定バックエンド`torchcodec`が
  FFmpeg共有ライブラリを要求し無かったため`soundfile`版へmonkeypatch(`python/infer.py`)。
- **未着手**: GPU(open-cuda/open-directx)推論、CPU命令セット活用(open-cpu、依存だけ追加済み)、
  `speech`重みの検証(basicのみ検証済み)。GUIは無い(HTTP API+ライブラリのみ)。

## 環境メモ(重要、次回の作業効率のため)

このセッションで判明した特性: **このマシン(F:\直下)ではPython(torch/transformers)の
importが極端に遅いことがある**(torch単体のimportで30〜40秒、transformers本体のimportが
90秒たっても終わらないケースを実機確認)。原因は不明(ディスクI/O自体は736MB/sと正常、
Windows Defenderのリアルタイム保護は無効化済みと確認済み)。**「遅いだけで実は進んでいる」
場合と「本当にスタックしている」場合の見分けが難しいため、プロセスを早期に強制終了しない
こと。** 判断材料: メモリ使用量が(遅くても)継続して増え続けていれば進行中、完全に横ばいが
続けば真にスタックしている可能性が高い。
