# P2 request classifier

The request classifier runs inside `nasiko-llm-router` at the same safe `cold_start`/`switch` boundaries as model-tier routing. It never runs during tool-loop continuation; the existing decision cache keeps a conversation on its selected model. The existing regex category classifier remains the default and needs no network.

## Backends

The trait returns `request_type`, `complexity` (1–5), and `confidence` (0–1). The regex implementation preserves the old category behavior and reports a neutral complexity of 3 and an uncalibrated confidence of 0.5. The hosted implementation sends the latest query plus up to six prior user/assistant turns (at most 4 KiB; tool outputs are omitted) to a configured OpenAI-compatible Chat Completions endpoint. The prompt treats both query and context as untrusted data, requests JSON, and accepts only known labels and in-range values. Invalid output fails closed.

`CLASSIFIER_BACKEND=hosted` opts into the remote backend. Requests use temperature 0, a fixed optional seed, a small completion limit, and a timeout. OpenRouter deployments use its standard `/api/v1/chat/completions` endpoint and optional app-attribution headers; see the [OpenRouter quickstart](https://openrouter.ai/docs/quickstart). Amazon Bedrock Mantle is also compatible with the configured base URL and model; see [AWS Chat Completions docs](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html). Supply endpoint, model, and API key through environment variables only. Do not commit credentials.

On timeout, transport/API error, malformed output, or confidence below `CLASSIFIER_MIN_CONFIDENCE`, the hosted wrapper returns the regex result and increments a process-local fallback counter. The router also validates third-party classifier results before using them. Confidence is a model-reported estimate, not a calibration guarantee; validate calibration and choose the threshold against held-out data before production use.

Complexity shifts the existing quality/cost blend: complexity 1 lowers the quality weight from 0.70 to 0.58, complexity 5 raises it to 0.82. Learned cells remain keyed by provider and request type. Thompson draws are seeded from `CLASSIFIER_TIER_SEED` plus provider, query, category, and complexity, so a fixed input and cell state produce repeatable tier decisions. Changing the seed is useful for sensitivity checks; this removes per-identical-input exploration variance. The selected model remains sticky through the existing cache.

## Configuration and deployment

For the Docker Compose deployment, add settings to the project-root `.env`; `docker-compose.yml` already passes that file into the server container. For source runs, use `server/.env`. Configuration is read at process startup, so restart the server after changing it. Example with OpenRouter:

```dotenv
CLASSIFIER_BACKEND=hosted
CLASSIFIER_ENDPOINT=https://openrouter.ai/api/v1
CLASSIFIER_MODEL=provider/model-id
CLASSIFIER_API_KEY=your-private-key
CLASSIFIER_TIMEOUT_MS=2500
CLASSIFIER_MIN_CONFIDENCE=0.55
CLASSIFIER_TIER_SEED=0
```

For the shared Bedrock Mantle setup, use base URL `https://bedrock-mantle.us-east-1.api.aws/v1`, model ID `openai.gpt-5.6-luna`, and put the shared Bedrock key in `CLASSIFIER_API_KEY` locally. The implementation appends `/chat/completions`; set the base URL only. Never paste the shared key into source, logs, or a PR. A Nasiko custom provider/config created in the UI controls an agent's normal LLM calls; the classifier's independent environment settings control this extra routing decision request.

The supplied `agents/*.zip` files are sample agent packages; no agent source changes are required for this router-only track. Any agent already sending LLM requests through Nasiko's router receives this behavior at the router boundary.

## Evaluation

```sh
EVAL_SET=llm-router/data/classifier_validation_v1.json OUT=/tmp/classifier-out.jsonl \
cargo run --release -p nasiko-llm-router --example classifier_eval
```

The JSONL output keeps the organizer contract (`id`, `request_type`, `complexity`, `confidence`, `latency_us`) and adds backend/fallback plus regex baseline fields. The example prints p50/p95 latency, fallback rate, request-type accuracy versus labels, regex accuracy, and complexity MAE to stderr. `latency_us` is per decision and excludes initialization. Repeat with `CLASSIFIER_BACKEND=hosted` and configured endpoint/model/key for a live comparison. Network latency and provider pricing vary; report the actual model, region/provider, measured latency, and cost from the provider's usage/pricing data. Do not claim cost savings from classifier accuracy alone.

Two author-labelled, disjoint datasets are included: `data/classifier_train_v1.json` (16 examples) and `data/classifier_validation_v1.json` (16 examples). Labels follow this rubric: classify the action the user asks for; pasted source code is `code_understanding` only when they ask to explain it, and code-shaped words inside quoted material do not imply a coding task. Mixed requests use the category of the principal requested work (code changes take precedence when the user asks to implement/fix); use `general` when no listed goal fits. Complexity 1 is a direct trivial task, 2 is a small task, 3 needs several steps or context, 4 has multiple connected requirements, and 5 needs deep analysis or many constraints. The split has no duplicate IDs or intentionally repeated prompts. Validation includes code-keyword traps, quoted text, mixed intent, context dominance, typos/noisy punctuation, code switching, underspecified and out-of-distribution requests, and multi-constraint reasoning. These are hand-authored analysis/evaluation examples; the hosted provider does not train on them or receive them as examples. Near-duplicate review is manual and should be repeated before adding cases.

## Limits

- The alternative backend implemented here is hosted only; a local model backend is not included.
- The upstream model's confidence is self-reported and may be miscalibrated.
- An OpenAI-compatible endpoint can still differ in model IDs or supported request fields; failures fall back to regex.
- Hosted mode sends the current request and bounded prior text to the configured provider. Leave the default regex mode enabled where that data flow is not acceptable.
- No downstream answer-quality/cost experiment is claimed; the evaluator compares classification labels and complexity only.
