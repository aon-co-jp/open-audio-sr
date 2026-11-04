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

## JSON処理の使用方針: `parse_secure_sonic`(速い型+安全)と`sonic-rs`は使える所では全て使う(2026-10-14、ユーザー指示)
正本は[`RS-JSON`](https://github.com/aon-co-jp/RS-JSON)(`README.md`・`PORTING.md`)。

**方針**: JSONを読む所は、使える所では**全て** `rust-json`(RS-JSON)の `parse_secure_sonic`(`fast`機能、結果は`sonic_rs::Value`)を使う。
`sonic-rs` を直接使える所(型付き構造体への読み込みなど)も同様。使えない所は `parse_secure`(結果は`serde_json::Value`)を使う。
外部から来る信頼できない入力(HTTP本文・ネットワーク越しのファイル・アップロード)は、**必ずサイズ・深さ・重複キーの上限つき**で読む。

**実測(7 MB・4万件)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms(約2.4倍速い、防御込み)** / `sonic-rs`そのまま 33 ms(約6倍、防御なし)。
`serde_json::Value`に読む限りSIMDでもほぼ速くならない(188 ms)。**速さの源は、結果を連続した領域に詰め込む値の型**。

**使える所 / 使えない所**
- ✅ 読み取り中心で、JSONを大きく・頻繁に読む所(キャッシュJSON、設定、APIの応答、ログ、インデックス)。
- ✅ 外部からの入力の入口(速さより防御が目的。`parse_secure`でも可)。
- ⚠️ `serde_json::Value`を関数間で受け渡している既存コードは、型を`sonic_rs::Value`に直す書き換えが要る(`JsonValueTrait`/`JsonContainerTrait`のメソッド)。書き換えずに`serde_json::Value`へ読むと速くならない。
- ❌ WASM(`rust_json::light`、依存ゼロ)、PHP/JavaScript側のJSON処理(Rustの部品は使えない)。
- 注意: 依存先の`sonic-rs`は内部で`unsafe`を使う(自前コードに`unsafe`は無い)。小さなJSON(数KB)では体感差が出ない。速度目的の書き換えは、実測で効く所から。

**導入**: `Cargo.toml`に `rust-json = { path = "../RS-JSON", features = ["fast"] }`。使用例は`RS-JSON/README.md`。

**このリポジトリでの適用状況**: 未適用(方針のみ)。コードの切り替えは、JSONを大きく・頻繁に読む所から実測して順に行う。

## JSON handling policy: use `parse_secure_sonic` (fast type + secure) and `sonic-rs` everywhere they can be used (2026-10-14, user instruction)
The source of truth is [`RS-JSON`](https://github.com/aon-co-jp/RS-JSON) (`README.md`, `PORTING.md`).

**Policy**: wherever JSON is read, use `parse_secure_sonic` from `rust-json` (RS-JSON, the `fast` feature; the result is `sonic_rs::Value`) **everywhere it can be used**.
Use `sonic-rs` directly where possible too (e.g. reading into typed structs). Where it cannot be used, use `parse_secure` (result: `serde_json::Value`).
Untrusted input from outside (HTTP bodies, files over the network, uploads) **must always be read with size / depth / duplicate-key limits**.

**Measured (7 MB, 40,000 records)**: `serde_json` 200 ms / `parse_secure` 217 ms / **`parse_secure_sonic` 83 ms (about 2.4x faster, with the defences)** / raw `sonic-rs` 33 ms (about 6x, no defences).
Reading into `serde_json::Value` gives almost no speed-up even with SIMD (188 ms). **The speed comes from the value type that packs the result into contiguous memory.**

**Where it can / cannot be used**
- ✅ Read-heavy code that parses large or frequent JSON (cache JSON, config, API responses, logs, indexes).
- ✅ Entrances for outside input (the goal there is defence rather than speed; `parse_secure` is fine).
- ⚠️ Existing code that passes `serde_json::Value` between functions needs its type changed to `sonic_rs::Value` (methods from `JsonValueTrait` / `JsonContainerTrait`). Reading into `serde_json::Value` without that change is not faster.
- ❌ WASM (`rust_json::light`, zero dependencies) and the JSON handling on the PHP / JavaScript side (Rust components cannot be used there).
- Note: the `sonic-rs` dependency uses `unsafe` internally (our own code has none). For small JSON (a few KB) the difference is not noticeable; do speed-motivated rewrites where measurements show a benefit.

**Adoption**: in `Cargo.toml`, `rust-json = { path = "../RS-JSON", features = ["fast"] }`. Example in `RS-JSON/README.md`.

**Status in this repository**: Not yet applied (policy only). Code will be switched over place by place, starting where measurements show large or frequent JSON reads.

---

作者 / Author: 石 塚 正 浩（MASAHIRO ISHIZUKA） (https://github.com/aon-co-jp)

## API 方針: REST 全廃・gRPC / Connect へ (2026-11-04)

### 日本語
- **REST は全廃する。新しい API は REST で作らない。** ブラウザや `curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`、本文は JSON)、サービス間は **gRPC / gRPC-web**(RPoem の管理API `open_runo.admin.v1.Admin` と同じ流儀)。ストリーミングは、必要になってから。
- **既存の REST は、触るときに順に移す。** 移したら旧 REST の経路は消す。移行が終わるまで残るが、**新規には追加しない**。
- **JSON は RS-JSON に統一**する(読み込みは `from_str_secure` / `parse_secure` でサイズ・深さ・重複キーの上限つき、書き出しは `rust_json::fast`)。ブラウザの WASM に `prost` は入れない(サイズが増えるため)。
- **例外(自分たちの API ではなく、相手の規格が決めているもの)**: ACME、OAuth/OIDC のコールバック、第三者(GitHub など)からの Webhook、ヘルスチェック、静的ファイル配信、OTLP など。これらは規格どおりの HTTP を受ける。
- 参考実装: `RPoem`(`open_runo.admin.v1.Admin`)、`open-easy-web`(`server/src/connect.rs`、`proto/open_easy_web/admin/v1/admin.proto`)。
- ※ このリポジトリのコードが既に方針どおりかどうかは、各リポジトリの現状による。この節は方針の記録であり、コードの変更を意味しない。

### English
- **REST is being abolished. New APIs are not built as REST.** APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`, JSON body); service-to-service calls use **gRPC / gRPC-web** (the same style as RPoem's admin API `open_runo.admin.v1.Admin`). Streaming only when it is actually needed.
- **Existing REST endpoints are migrated one by one, when they are touched.** The old REST route is removed once migrated. Until migration is finished they remain, but **no new ones are added**.
- **JSON is unified on RS-JSON**: reading with `from_str_secure` / `parse_secure` (size, depth and duplicate-key limits), writing with `rust_json::fast`. `prost` is not put into browser WASM (it would grow the binary).
- **Exceptions (shapes dictated by someone else's standard, not our own API)**: ACME, OAuth/OIDC callbacks, webhooks from third parties (e.g. GitHub), health checks, static file serving, OTLP, and so on. These accept plain HTTP as the standard requires.
- Reference implementations: `RPoem` (`open_runo.admin.v1.Admin`) and `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Note: whether the code in a given repository already follows this policy depends on that repository. This section records the policy; it does not mean the code has been changed.

### Deutsch (KI-Übersetzung, redaktionell überarbeitet; noch nicht von Muttersprachlern geprüft)
- **REST wird vollständig abgeschafft; neue APIs entstehen nicht mehr als REST.** APIs, die aus dem Browser oder per `curl` aufgerufen werden, verwenden **unäre Connect-Aufrufe** (`POST /<Paket>.<Service>/<Methode>`, `content-type: application/json`, JSON-Body). Zwischen Diensten kommen **gRPC bzw. gRPC-web** zum Einsatz, im selben Stil wie die Admin-API von RPoem (`open_runo.admin.v1.Admin`). Streaming wird erst ergänzt, wenn es tatsächlich gebraucht wird.
- **Bestehende REST-Endpunkte werden schrittweise umgestellt, und zwar immer dann, wenn man ohnehin an ihnen arbeitet.** Sobald ein Endpunkt umgestellt ist, wird die alte REST-Route entfernt. Bis zum Abschluss der Migration bleiben die übrigen bestehen, es werden aber **keine neuen mehr hinzugefügt**.
- **JSON wird einheitlich über RS-JSON verarbeitet**: Gelesen wird mit `from_str_secure` bzw. `parse_secure` (mit Obergrenzen für Größe, Verschachtelungstiefe und doppelte Schlüssel), geschrieben mit `rust_json::fast`. `prost` wird nicht in das Browser-WASM aufgenommen, da es die Datei vergrößern würde.
- **Ausnahmen (Schnittstellen, deren Form ein fremder Standard vorgibt und die nicht zu unserer eigenen API gehören)**: ACME, OAuth/OIDC-Callbacks, Webhooks von Drittanbietern (z. B. GitHub), Health-Checks, die Auslieferung statischer Dateien, OTLP und Ähnliches. Diese nehmen, wie vom jeweiligen Standard verlangt, gewöhnliches HTTP entgegen.
- Referenzimplementierungen: `RPoem` (`open_runo.admin.v1.Admin`) und `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Hinweis: Ob der Code eines einzelnen Repositorys diese Richtlinie bereits erfüllt, hängt vom jeweiligen Repository ab. Dieser Abschnitt dokumentiert lediglich die Richtlinie; er besagt nicht, dass der Code schon angepasst wurde.

### Русский (перевод выполнен с помощью ИИ и отредактирован; носителями языка пока не проверен)
- **От REST мы полностью отказываемся; новые API больше не создаются в стиле REST.** API, к которым обращаются из браузера или через `curl`, используют **унарные вызовы Connect** (`POST /<пакет>.<Service>/<Method>`, `content-type: application/json`, тело запроса — JSON). Для взаимодействия между сервисами применяются **gRPC и gRPC-web** — в том же стиле, что и административный API RPoem (`open_runo.admin.v1.Admin`). Потоковая передача данных добавляется лишь тогда, когда в ней действительно возникает необходимость.
- **Существующие REST-эндпоинты переводятся поэтапно — по мере того, как с ними приходится работать.** После перевода старый REST-маршрут удаляется. Пока миграция не завершена, оставшиеся эндпоинты сохраняются, но **новые не добавляются**.
- **Работа с JSON унифицирована на основе RS-JSON**: чтение — через `from_str_secure` или `parse_secure` (с ограничениями на размер, глубину вложенности и повторяющиеся ключи), запись — через `rust_json::fast`. `prost` в браузерный WASM не включается, поскольку это увеличило бы размер файла.
- **Исключения (интерфейсы, форму которых определяет чужой стандарт, а не наш собственный API)**: ACME, обратные вызовы OAuth/OIDC, вебхуки сторонних сервисов (например, GitHub), проверки работоспособности, раздача статических файлов, OTLP и тому подобное. Такие интерфейсы принимают обычный HTTP, как того требует соответствующий стандарт.
- Эталонные реализации: `RPoem` (`open_runo.admin.v1.Admin`) и `open-easy-web` (`server/src/connect.rs`, `proto/open_easy_web/admin/v1/admin.proto`).
- Примечание: соответствует ли код конкретного репозитория этой политике, зависит от самого репозитория. Данный раздел лишь фиксирует политику и не означает, что код уже изменён.
