# AI CBT Evaluation Suite

This repository runs a controlled comparison of baseline LLM outputs versus CBT-style (cognitive-behavioral) reflective revision across multiple task suites, with optional evaluator models and a Streamlit dashboard for inspection.

## Purpose
- Measure whether a CBT-inspired reflection + revision loop improves clarity, coherence, reasoning depth, safety, and overall quality versus baseline outputs.
- Support both single-turn tasks (simple + advanced) and multi-round longitudinal tasks.
- Provide evaluator model scoring and a visual dashboard to compare models head-to-head.

## Repository Layout
- `run_experiment.py` — single-turn runner (simple + advanced tasks) for baseline vs CBT; writes results and evaluator scores.
- `run_longitudinal.py` — multi-round runner for longitudinal tasks under baseline vs CBT.
- `app.py` — Streamlit dashboard to browse outputs, reflections, scores, and longitudinal histories.
- `core/llm_client.py` — OpenRouter chat client with retries and logging.
- `core/cbt_agent.py` — CBT meta-agent that detects distortions and issues revision prompts.
- `core/experiment_runner.py` — orchestrates single-turn baseline/CBT runs per model and task.
- `core/longitudinal_runner.py` — orchestrates multi-round baseline/CBT runs per model and task.
- `core/evaluator.py` — scores baseline vs CBT revised outputs using evaluator models.
- `core/task_loader.py` — YAML task loader.
- `config/models.yaml` — model configs for client, CBT meta-agent, and evaluators.
- `config/tasks_simple.yaml`, `config/tasks_advanced.yaml`, `config/tasks_longitudinal.yaml` — task suites.
- `output/` — results and logs are written here (JSON + timestamped logs).
- `.env` — holds `OPENROUTER_API_KEY`.
- `requirements.txt` — Python dependencies.

## How the Experiment Works (Data Flow)
1. **Config load**: `run_experiment.py` or `run_longitudinal.py` loads `config/models.yaml` and the task YAMLs via `core/task_loader.py`.
2. **Client model selection**: For each `client_model` entry, a `core/llm_client.LLMClient` is built to call the OpenRouter API.
3. **Baseline condition**:
   - Single-turn: `core/experiment_runner.py` sends the task prompt directly to the client model and records the response as baseline.
   - Longitudinal: `core/longitudinal_runner.py` iterates rounds; each round feeds the previous output into the task’s `round_prompt_template` (baseline has no CBT reflection).
4. **CBT condition**:
   - A CBT meta-agent (`core/cbt_agent.CBTAgent`) built with the `cbt_model` from `config/models.yaml` analyzes the client’s raw output using `CBT_DETECTION_PROMPT`.
   - The CBT agent returns JSON (`distortions`, `explanation`, `guidance`, `revision_instruction`). If JSON is malformed, it self-repairs.
   - `revision_instruction` is wrapped via `CBT_REVISION_WRAPPER` and sent back to the same client model to produce the revised answer.
   - Longitudinal CBT uses the revised answer as the “previous” text for the next round.
5. **Evaluation** (single-turn only): `core/evaluator.py` prompts each evaluator model to compare baseline vs CBT revised output and return JSON scores (clarity, coherence, reasoning_depth, safety, overall + comment).
6. **Persistence**:
   - Single-turn results → `output/single_turn_results.json`
   - Longitudinal results → `output/longitudinal_results.json`
   - Logs → `output/experiment_*.log`, `output/longitudinal_*.log`
7. **Visualization**: `app.py` (Streamlit) reads the JSONs and displays per-task/model comparisons, reflections, evaluator scores, and longitudinal histories side by side.

## Task Suites
- `config/tasks_simple.yaml`: 5 short-form prompts.
- `config/tasks_advanced.yaml`: 5 more complex prompts.
- `config/tasks_longitudinal.yaml`: 5 multi-round scenarios with `round_prompt_template` placeholders.

## Model Configuration (`config/models.yaml`)
- `client_models`: models being tested (baseline vs CBT). Each has `id`, `name` (OpenRouter model name), `temperature`, `max_tokens`.
- `cbt_model`: single model for the meta-agent reflection.
- `evaluator_models`: models that score CBT vs baseline (single-turn only).
Adjust IDs/names to match available OpenRouter models. Temperatures and token limits can be tuned per model.

## Logging
- Each run emits a timestamped log file in `output/` (e.g., `experiment_YYYYMMDD_HHMMSS.log` or `longitudinal_YYYYMMDD_HHMMSS.log`) plus console output.
- LLM call attempts, CBT parsing/repair warnings, and evaluator JSON parsing warnings are logged.

## Running the Experiments
1. Set the API key in `.env`:
   - `OPENROUTER_API_KEY="your-key"`
2. Install deps:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Single-turn (simple + advanced):
   ```bash
   python3 run_experiment.py
   ```
   - Outputs: `output/single_turn_results.json`, `output/experiment_*.log`
4. Longitudinal:
   ```bash
   python3 run_longitudinal.py
   ```
   - Outputs: `output/longitudinal_results.json`, `output/longitudinal_*.log`
5. Dashboard (after results exist):
   ```bash
   streamlit run app.py
   ```

## Output Schemas (high level)
- `single_turn_results.json` (list of records):
  - `model_id`, `model_name`, `task_id`, `task_prompt`
  - `condition_results.baseline`: baseline text
  - `condition_results.cbt.raw`: initial answer before CBT
  - `condition_results.cbt.reflection`: CBT JSON (distortions, guidance, revision_instruction, etc.)
  - `condition_results.cbt.revised`: revised answer after CBT prompt
  - `evaluation`: list of evaluator outputs with `score_parsed` (JSON scores) and `score_raw`
- `longitudinal_results.json` (per model):
  - `tasks[task_id].task_prompt`
  - `baseline_history`: list of rounds (`round`, `prompt`, `raw`)
  - `cbt_history`: list of rounds with `raw`, `reflection`, `revision_prompt`, `revised`

## How the CBT Loop Works
- Detection prompt: `core/cbt_agent.py` → `CBT_DETECTION_PROMPT` asks the CBT model to find distortions and produce a revision instruction in JSON.
- Repair: If the CBT model’s output is not valid JSON, a repair prompt is issued to produce clean JSON.
- Revision: `CBT_REVISION_WRAPPER` injects the revision instruction and asks the original client model to produce a corrected answer, emphasizing nuance, grounding, and reduced distortions.
- Longitudinal: The revised answer is fed into the next-round prompt template to iteratively improve over multiple rounds.

### CBT “therapy” content (both single-turn and longitudinal)
- **Detection prompt** (`core/cbt_agent.py:CBT_DETECTION_PROMPT`): The CBT model reviews the client model’s answer, detects distortions (overgeneralization, black-and-white thinking, overconfidence, hallucinated facts, missing caveats, contradictions, shallow explanations, inappropriate emotional tone), and returns pure JSON with `distortions`, `explanation`, `guidance`, and a `revision_instruction`.
- **Revision wrapper** (`core/cbt_agent.py:CBT_REVISION_WRAPPER`): The revision instruction is quoted and sent back to the same client model, asking it to revise the prior answer to reduce distortions, add nuance/grounding, avoid hallucinations/overconfidence, and remain clear and structured. The same CBT loop is used for both single-turn tasks and each round of longitudinal CBT condition.

## Evaluations
- `core/evaluator.py` compares baseline vs CBT revised answers using each evaluator model.
- Scoring criteria: clarity, coherence, reasoning_depth, safety, overall (1–10) plus a short comment.
- Parsing: If JSON parsing fails, the raw response is still stored for inspection.

## Dashboard Usage (`app.py`)
- Single-turn view:
  - Select model and task; see baseline, CBT raw, CBT reflection JSON, and CBT revised text.
  - Evaluator scores table and aggregated averages.
  - Cross-model comparison pivot (averaged scores per evaluator/model).
- Longitudinal view:
  - Select model and task; expand per-round baseline and CBT histories (prompts, raw outputs, reflections, revisions).

## Extending or Modifying
- Add/remove tasks: edit `config/tasks_simple.yaml`, `config/tasks_advanced.yaml`, `config/tasks_longitudinal.yaml`.
- Add models: update `config/models.yaml` with new `client_models`, change `cbt_model`, and/or adjust `evaluator_models`.
- Tuning: adjust `temperature`, `max_tokens` in `config/models.yaml`.
- Rounds: change `rounds` in `run_longitudinal.py`.
- Logging level: modify `logging.basicConfig` in the entry scripts.

## Troubleshooting
- Key errors when formatting CBT prompt: fixed by escaping braces in `core/cbt_agent.py`.
- Non-JSON evaluator/CBT outputs: check logs; raw text is preserved, and CBT auto-repair attempts JSON fix.
- API failures/throttling: LLM client retries up to 3 times; see logs in `output/`.

## Notes on Keys and Secrets
- Keep `.env` out of version control. Only `OPENROUTER_API_KEY` is required for these runs.
