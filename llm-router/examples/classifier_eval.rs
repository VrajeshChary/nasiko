//! Request classifier evaluation.
//!
//! Run the offline baseline:
//!   EVAL_SET=/tmp/classifier-eval.json OUT=/tmp/classifier-out.jsonl \
//!   cargo run --release -p nasiko-llm-router --example classifier_eval
//!
//! Opt in to an OpenAI-compatible classifier with CLASSIFIER_BACKEND=hosted plus
//! CLASSIFIER_ENDPOINT, CLASSIFIER_MODEL and (if required) CLASSIFIER_API_KEY.
//! Per-case latency excludes startup/client initialization.
use std::io::Write;
use std::time::Instant;

use nasiko_llm_router::config::ClassifierConfig;
use nasiko_llm_router::routing::{
    ClassifyInput, RegexClassifier, RequestClassifier, build_request_classifier,
};

#[tokio::main]
async fn main() {
    let path = std::env::var("EVAL_SET").expect("set EVAL_SET to the eval JSON path");
    let out_path = std::env::var("OUT").unwrap_or_else(|_| "classifier-out.jsonl".into());
    let raw = std::fs::read_to_string(&path).expect("read EVAL_SET");
    let data: serde_json::Value = serde_json::from_str(&raw).expect("valid eval JSON");
    let examples = data["examples"].as_array().expect("examples array");

    let settings = ClassifierConfig::from_env();
    let http = reqwest::Client::new();
    let classifier = build_request_classifier(&settings, http);
    let regex = RegexClassifier;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create OUT"));
    let mut latencies = Vec::with_capacity(examples.len());
    let (mut model_correct, mut regex_correct, mut complexity_error, mut scored) =
        (0usize, 0usize, 0u64, 0usize);
    for example in examples {
        let id = example["id"].as_str().expect("id");
        let query = example["query"].as_str().expect("query");
        let context = example["context"].as_str();
        let expected_type = example["request_type"].as_str();
        let expected_complexity = example["complexity"].as_u64();
        let input = ClassifyInput { query, context };

        let baseline = regex
            .classify(&input)
            .await
            .expect("regex classifier is infallible");
        let before_fallbacks = classifier.fallback_count();
        let started = Instant::now();
        let result = match classifier.classify(&input).await {
            Ok(result) => result,
            Err(error) => {
                eprintln!(
                    "classifier {} failed on {id}: {error}; using regex",
                    classifier.name()
                );
                baseline
            }
        };
        let latency_us = started.elapsed().as_micros() as u64;
        latencies.push(latency_us);
        let request_type_correct =
            expected_type.map(|expected| result.request_type.as_str() == expected);
        let complexity_correct = expected_complexity
            .and_then(|expected| u8::try_from(expected).ok())
            .map(|expected| result.complexity == expected);
        if let (Some(expected_type), Some(expected_complexity)) = (
            expected_type,
            expected_complexity.and_then(|n| u8::try_from(n).ok()),
        ) {
            scored += 1;
            model_correct += if result.request_type.as_str() == expected_type {
                1
            } else {
                0
            };
            regex_correct += if baseline.request_type.as_str() == expected_type {
                1
            } else {
                0
            };
            complexity_error += u64::from(result.complexity.abs_diff(expected_complexity));
        }
        let line = serde_json::json!({
            "id": id,
            "request_type": result.request_type.as_str(),
            "complexity": result.complexity,
            "confidence": result.confidence,
            "expected_request_type": expected_type,
            "expected_complexity": expected_complexity,
            "request_type_correct": request_type_correct,
            "complexity_correct": complexity_correct,
            "latency_us": latency_us,
            "classifier": classifier.name(),
            "fallback": classifier.fallback_count() > before_fallbacks,
            "fallback_count": classifier.fallback_count(),
            "regex_baseline": {
                "request_type": baseline.request_type.as_str(),
                "complexity": baseline.complexity,
                "confidence": baseline.confidence,
            }
        });
        writeln!(out, "{line}").expect("write OUT");
    }
    out.flush().expect("flush OUT");
    if !latencies.is_empty() {
        latencies.sort_unstable();
        let percentile =
            |p: usize| latencies[((latencies.len() - 1) * p / 100).min(latencies.len() - 1)];
        let fallback_rate = classifier.fallback_count() as f64 / examples.len() as f64;
        eprintln!(
            "classifier={} cases={} p50_us={} p95_us={} fallback_rate={:.3}",
            classifier.name(),
            examples.len(),
            percentile(50),
            percentile(95),
            fallback_rate
        );
        if scored > 0 {
            eprintln!(
                "labelled_cases={} request_type_accuracy={:.3} regex_accuracy={:.3} complexity_mae={:.3}",
                scored,
                model_correct as f64 / scored as f64,
                regex_correct as f64 / scored as f64,
                complexity_error as f64 / scored as f64
            );
        }
    }
}
