//! open-audio-sr: AudioSR(実在する帯域拡張AIモデル)をRPoem(`open-runo-poem-compat`)上で
//! HTTP APIとして提供する。他プロジェクトから同一プロセス内のライブラリ呼び出し(`open_audio_sr::super_resolve_file`)
//! でも、ネットワーク越しのHTTP呼び出しでも使えるように、両方とも同じ関数を経由する。
//!
//! - `POST /v1/super_resolve`(bodyは音声ファイルのバイト列、`?model=basic&guidance_scale=3.5&ddim_steps=10`)
//!   → 帯域拡張後のWAV(48kHz)をバイト列で返す
//! - `GET /healthz` / `GET /v1/health`(Python venv・推論スクリプトが揃っているかの軽い確認)

use std::path::PathBuf;
use std::sync::Arc;

use http_body_util::{BodyExt, Limited};
use open_runo_poem_compat::hyper_compat::{fixed_body, json_response};
use open_runo_poem_compat::{get, handler_fn, post, Request, Response, Route, Server, StatusCode, TcpListener};
use serde_json::json;

use open_audio_sr::{super_resolve_file, SrOptions};

struct Ctx {
    tmp_dir: PathBuf,
}

fn err_json(status: StatusCode, msg: impl Into<String>) -> Response {
    json_response(status, &json!({ "error": msg.into() }))
}

fn parse_query(req: &Request) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    if let Some(q) = req.uri().query() {
        for pair in q.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                map.insert(k.to_string(), v.to_string());
            }
        }
    }
    map
}

fn app(ctx: Arc<Ctx>) -> Route {
    let c1 = ctx.clone();
    Route::new()
        .at("/healthz", get(handler_fn(|_r, _p| async { json_response(StatusCode::OK, &json!({ "ok": true })) })))
        .at(
            "/v1/health",
            get(handler_fn(|_r, _p| async {
                json_response(StatusCode::OK, &json!({ "available": open_audio_sr::is_available() }))
            })),
        )
        .at(
            "/v1/super_resolve",
            post(handler_fn(move |req: Request, _p| {
                let ctx = c1.clone();
                async move {
                    let q = parse_query(&req);
                    // 64MB上限(数分の音声を想定)。それ以上は別途チャンク分割や
                    // ストリーミングAPIが必要になるが、現状のスコープ外。
                    let Ok(body) = Limited::new(req.into_body(), 64 << 20).collect().await else {
                        return err_json(StatusCode::PAYLOAD_TOO_LARGE, "音声が大きすぎます(上限64MB)");
                    };
                    let bytes = body.to_bytes();
                    if bytes.is_empty() {
                        return err_json(StatusCode::BAD_REQUEST, "音声データが空です");
                    }
                    let opts = SrOptions {
                        guidance_scale: q.get("guidance_scale").and_then(|v| v.parse().ok()).unwrap_or(3.5),
                        ddim_steps: q.get("ddim_steps").and_then(|v| v.parse().ok()).unwrap_or(10),
                        seed: q.get("seed").and_then(|v| v.parse().ok()).unwrap_or(42),
                        model_name: q.get("model").cloned().unwrap_or_else(|| "basic".to_string()),
                    };
                    let id = uuid_like();
                    let input = ctx.tmp_dir.join(format!("{id}_in.wav"));
                    let output = ctx.tmp_dir.join(format!("{id}_out.wav"));
                    if let Err(e) = std::fs::write(&input, &bytes) {
                        return err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("一時ファイルへ書き込めません: {e}"));
                    }
                    let input_for_task = input.clone();
                    let output_for_task = output.clone();
                    let result = tokio::task::spawn_blocking(move || super_resolve_file(&input_for_task, &output_for_task, &opts)).await;
                    let _ = std::fs::remove_file(&input);
                    match result {
                        Ok(Ok(())) => match std::fs::read(&output) {
                            Ok(wav) => {
                                let _ = std::fs::remove_file(&output);
                                hyper::Response::builder().status(StatusCode::OK).header("content-type", "audio/wav").body(fixed_body(wav.into())).unwrap()
                            }
                            Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("出力を読めません: {e}")),
                        },
                        Ok(Err(e)) => err_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
                        Err(e) => err_json(StatusCode::INTERNAL_SERVER_ERROR, format!("推論タスクが異常終了しました: {e}")),
                    }
                }
            })),
        )
}

/// 依存を増やさないための簡易な一意名(プロセスID+単調増加カウンタ)。UUIDの代わり。
fn uuid_like() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    format!("{}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed))
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let bind: std::net::SocketAddr = std::env::var("OPEN_AUDIO_SR_BIND").unwrap_or_else(|_| "127.0.0.1:4620".into()).parse().expect("OPEN_AUDIO_SR_BIND は ホスト:ポート 形式で指定してください");
    let tmp_dir = std::env::temp_dir().join("open_audio_sr");
    std::fs::create_dir_all(&tmp_dir)?;
    if !open_audio_sr::is_available() {
        eprintln!("open-audio-sr: 警告: Python venv または python/infer.py が見つかりません。README.mdのセットアップ手順を確認してください。");
    }
    let ctx = Arc::new(Ctx { tmp_dir });
    let (addr, handle) = Server::new(TcpListener::bind(bind)).run(app(ctx)).await?;
    println!("open-audio-sr: http://{addr}/ で待受中(POST /v1/super_resolve)");
    tokio::select! {
        _ = handle => {}
        _ = tokio::signal::ctrl_c() => println!("open-audio-sr: 終了します"),
    }
    Ok(())
}
