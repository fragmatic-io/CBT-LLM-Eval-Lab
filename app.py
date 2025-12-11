import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load {path}: {exc}")
        return None


def build_eval_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        evals = rec.get("evaluation", []) or []
        for ev in evals:
            parsed = ev.get("score_parsed") or {}
            overall = parsed.get("overall")
            rows.append(
                {
                    "model_id": rec.get("model_id"),
                    "model_name": rec.get("model_name"),
                    "task_id": rec.get("task_id"),
                    "evaluator": ev.get("evaluator_model_id"),
                    "clarity": parsed.get("clarity"),
                    "coherence": parsed.get("coherence"),
                    "reasoning_depth": parsed.get("reasoning_depth"),
                    "safety": parsed.get("safety"),
                    "overall": overall,
                    "lift_pct": (overall / 10 * 100) if overall is not None else None,
                    "comment": parsed.get("comment") or ev.get("score_raw"),
                }
            )
    return pd.DataFrame(rows)


def build_long_eval_df(long_data: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_id, rec in long_data.items():
        model_name = rec.get("model_name")
        for task_id, task_rec in rec.get("tasks", {}).items():
            for ev in task_rec.get("final_evaluation", []) or []:
                parsed = ev.get("score_parsed") or {}
                overall = parsed.get("overall")
                rows.append(
                    {
                        "model_id": model_id,
                        "model_name": model_name,
                        "task_id": task_id,
                        "evaluator": ev.get("evaluator_model_id"),
                        "clarity": parsed.get("clarity"),
                        "coherence": parsed.get("coherence"),
                        "reasoning_depth": parsed.get("reasoning_depth"),
                        "safety": parsed.get("safety"),
                        "overall": overall,
                        "lift_pct": (overall / 10 * 100) if overall is not None else None,
                        "comment": parsed.get("comment") or ev.get("score_raw"),
                    }
                )
    return pd.DataFrame(rows)


def render_single_turn(single_data: List[Dict[str, Any]]) -> None:
    st.subheader("Single-Turn Tasks (Baseline vs CBT)")
    if not single_data:
        st.info("No single_turn_results.json found in output/. Run run_experiment.py first.")
        return

    eval_df_all = build_eval_df(single_data)
    st.markdown("### Overall CBT vs Baseline (across all models and tasks)")
    if eval_df_all.empty:
        st.info("No evaluator outputs yet.")
    else:
        overall_mean = eval_df_all["overall"].mean()
        lift_pct_mean = eval_df_all["lift_pct"].mean()
        col_a, col_b = st.columns(2)
        col_a.metric("Avg relative overall (0-10)", f"{overall_mean:.2f}")
        col_b.metric("Avg lift (% of max)", f"{lift_pct_mean:.1f}%")

        summary = (
            eval_df_all.groupby("model_id")[["overall", "clarity", "coherence", "reasoning_depth", "safety", "lift_pct"]]
            .mean()
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        chart = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                x=alt.X("model_id:N", title="Model"),
                y=alt.Y("overall:Q", title="Avg Overall Score (CBT vs baseline)"),
                tooltip=["model_id", "overall", "clarity", "coherence", "reasoning_depth", "safety"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        by_task = (
            eval_df_all.groupby("task_id")[["overall", "lift_pct"]]
            .mean()
            .reset_index()
            .sort_values("overall", ascending=False)
        )
        st.markdown("**Average overall by task (CBT vs baseline)**")
        st.dataframe(by_task, use_container_width=True)

        st.markdown("**Heatmap: Model × Task (overall score)**")
        model_task = (
            eval_df_all.groupby(["model_id", "task_id"])[["overall", "lift_pct"]]
            .mean()
            .reset_index()
        )
        heatmap = (
            alt.Chart(model_task)
            .mark_rect()
            .encode(
                x=alt.X("task_id:N", title="Task"),
                y=alt.Y("model_id:N", title="Model"),
                color=alt.Color("overall:Q", title="Overall", scale=alt.Scale(scheme="blues")),
                tooltip=["model_id", "task_id", "overall"],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)

        st.markdown("**Distribution of overall scores (all models, tasks, evaluators)**")
        box = (
            alt.Chart(eval_df_all)
            .mark_boxplot()
            .encode(x=alt.X("model_id:N", title="Model"), y=alt.Y("overall:Q"))
        )
        st.altair_chart(box, use_container_width=True)

        st.markdown("**Average overall by evaluator**")
        by_eval = (
            eval_df_all.groupby("evaluator")[["overall"]]
            .mean()
            .reset_index()
            .sort_values("overall", ascending=False)
        )
        st.dataframe(by_eval, use_container_width=True)
        eval_bar = (
            alt.Chart(by_eval)
            .mark_bar()
            .encode(x=alt.X("evaluator:N", title="Evaluator"), y=alt.Y("overall:Q"))
        )
        st.altair_chart(eval_bar, use_container_width=True)

        st.markdown("### Delta view (CBT lift vs baseline)")
        # Scores are already relative (CBT vs baseline), so treat them as the delta
        delta_model = (
            eval_df_all.groupby("model_id")[["overall", "lift_pct"]]
            .mean()
            .reset_index()
            .sort_values("overall", ascending=False)
        )
        st.markdown("**CBT lift by model (mean relative overall score and % of max)**")
        st.dataframe(delta_model, use_container_width=True)
        delta_chart = (
            alt.Chart(delta_model)
            .mark_bar()
            .encode(
                x=alt.X("model_id:N", title="Model"),
                y=alt.Y("overall:Q", title="Relative overall (CBT vs baseline)"),
                tooltip=["model_id", "overall", "lift_pct"],
            )
        )
        st.altair_chart(delta_chart, use_container_width=True)

        delta_task = (
            eval_df_all.groupby("task_id")[["overall", "lift_pct"]]
            .mean()
            .reset_index()
            .sort_values("overall", ascending=False)
        )
        st.markdown("**CBT lift by task (mean relative overall score and % of max)**")
        st.dataframe(delta_task, use_container_width=True)
        delta_task_chart = (
            alt.Chart(delta_task)
            .mark_bar()
            .encode(
                x=alt.X("task_id:N", title="Task"),
                y=alt.Y("overall:Q", title="Relative overall (CBT vs baseline)"),
                tooltip=["task_id", "overall", "lift_pct"],
            )
        )
        st.altair_chart(delta_task_chart, use_container_width=True)

    model_ids = sorted({r["model_id"] for r in single_data})
    task_ids = sorted({r["task_id"] for r in single_data})

    col1, col2 = st.columns(2)
    model_choice = col1.selectbox("Model", model_ids)
    task_choice = col2.selectbox("Task", task_ids)

    filtered = [
        r for r in single_data if r["model_id"] == model_choice and r["task_id"] == task_choice
    ]
    if not filtered:
        st.warning("No matching record.")
        return

    rec = filtered[0]
    st.write(f"**Task Prompt:** {rec['task_prompt']}")

    tabs = st.tabs(["Baseline", "CBT Raw", "CBT Reflection", "CBT Revised"])
    baseline = rec["condition_results"]["baseline"]
    cbt_raw = rec["condition_results"]["cbt"]["raw"]
    reflection = rec["condition_results"]["cbt"]["reflection"]
    revised = rec["condition_results"]["cbt"]["revised"]

    with tabs[0]:
        st.write(baseline)
    with tabs[1]:
        st.write(cbt_raw)
    with tabs[2]:
        st.json(reflection, expanded=True)
    with tabs[3]:
        st.write(revised)

    st.markdown("---")
    st.markdown("**Evaluator Scores**")
    eval_df = build_eval_df(filtered)
    if eval_df.empty:
        st.info("No evaluator outputs yet.")
    else:
        st.dataframe(eval_df, use_container_width=True)
        agg = eval_df.groupby("model_id")[["clarity", "coherence", "reasoning_depth", "safety", "overall"]].mean().reset_index()
        st.markdown("**Average Scores (selected model across this task)**")
        st.dataframe(agg, use_container_width=True)

    st.markdown("---")
    st.markdown("**Cross-Model Comparison (overall scores)**")
    full_eval = build_eval_df(single_data)
    if full_eval.empty:
        st.info("No evaluation data to compare yet.")
    else:
        pivot = (
            full_eval.groupby(["model_id", "evaluator"])[["overall", "clarity", "coherence", "reasoning_depth", "safety"]]
            .mean()
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)


def render_longitudinal(long_data: Dict[str, Any]) -> None:
    st.subheader("Longitudinal Tasks (multi-round baseline vs CBT)")
    if not long_data:
        st.info("No longitudinal_results.json found in output/. Run run_longitudinal.py first.")
        return

    eval_df_all = build_long_eval_df(long_data)
    st.markdown("### Overall CBT vs Baseline (final round comparison)")
    if eval_df_all.empty:
        st.info("No evaluator outputs yet.")
    else:
        overall_mean = eval_df_all["overall"].mean()
        lift_pct_mean = eval_df_all["lift_pct"].mean()
        col_a, col_b = st.columns(2)
        col_a.metric("Avg relative overall (0-10)", f"{overall_mean:.2f}")
        col_b.metric("Avg lift (% of max)", f"{lift_pct_mean:.1f}%")

        summary = (
            eval_df_all.groupby("model_id")[["overall", "clarity", "coherence", "reasoning_depth", "safety", "lift_pct"]]
            .mean()
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True)

        chart = (
            alt.Chart(summary)
            .mark_bar()
            .encode(
                x=alt.X("model_id:N", title="Model"),
                y=alt.Y("overall:Q", title="Avg Overall Score"),
                tooltip=["model_id", "overall", "clarity", "coherence", "reasoning_depth", "safety"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Delta view (CBT lift vs baseline, final round)")
        delta_model = (
            eval_df_all.groupby("model_id")[["overall"]]
            .mean()
            .reset_index()
            .sort_values("overall", ascending=False)
        )
        st.dataframe(delta_model, use_container_width=True)
        delta_chart = (
            alt.Chart(delta_model)
            .mark_bar()
            .encode(
                x=alt.X("model_id:N", title="Model"),
                y=alt.Y("overall:Q", title="Relative overall (CBT vs baseline)"),
                tooltip=["model_id", "overall"],
            )
        )
        st.altair_chart(delta_chart, use_container_width=True)

        st.markdown("**Heatmap: Model × Task (overall score)**")
        model_task = (
            eval_df_all.groupby(["model_id", "task_id"])[["overall", "lift_pct"]]
            .mean()
            .reset_index()
        )
        heatmap = (
            alt.Chart(model_task)
            .mark_rect()
            .encode(
                x=alt.X("task_id:N", title="Task"),
                y=alt.Y("model_id:N", title="Model"),
                color=alt.Color("overall:Q", scale=alt.Scale(scheme="greens")),
                tooltip=["model_id", "task_id", "overall"],
            )
        )
        st.altair_chart(heatmap, use_container_width=True)

    model_ids = sorted(long_data.keys())
    model_choice = st.selectbox("Model", model_ids)

    task_ids = sorted(long_data[model_choice]["tasks"].keys())
    task_choice = st.selectbox("Task", task_ids)

    task_rec = long_data[model_choice]["tasks"][task_choice]
    st.write(f"**Task Prompt:** {task_rec['task_prompt']}")

    baseline_hist = task_rec["baseline_history"]
    cbt_hist = task_rec["cbt_history"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline rounds**")
        for round_rec in baseline_hist:
            with st.expander(f"Round {round_rec['round']}"):
                st.write(round_rec["raw"])

    with col2:
        st.markdown("**CBT rounds**")
        for round_rec in cbt_hist:
            with st.expander(f"Round {round_rec['round']}"):
                st.write("Prompt:", round_rec["prompt"])
                st.write("Raw:", round_rec["raw"])
                st.json(round_rec.get("reflection", {}), expanded=False)
                st.write("Revised:", round_rec.get("revised", ""))

    st.markdown("---")
    st.markdown("**Final Round Evaluation (Baseline vs CBT)**")
    eval_scores = task_rec.get("final_evaluation") or []
    if not eval_scores:
        st.info("No evaluator scores for longitudinal tasks yet.")
    else:
        rows = []
        for ev in eval_scores:
            parsed = ev.get("score_parsed") or {}
            rows.append(
                {
                    "evaluator": ev.get("evaluator_model_id"),
                    "clarity": parsed.get("clarity"),
                    "coherence": parsed.get("coherence"),
                    "reasoning_depth": parsed.get("reasoning_depth"),
                    "safety": parsed.get("safety"),
                    "overall": parsed.get("overall"),
                    "comment": parsed.get("comment") or ev.get("score_raw"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main() -> None:
    st.title("AI CBT Evaluation Dashboard")
    st.write(
        "Visualize baseline vs CBT outputs, evaluator scores, and longitudinal runs. "
        "Ensure you have run the experiments to populate JSON files in output/."
    )

    single_turn_path = OUTPUT_DIR / "single_turn_results.json"
    longitudinal_path = OUTPUT_DIR / "longitudinal_results.json"

    single_data = load_json(single_turn_path)
    long_data = load_json(longitudinal_path)

    view = st.sidebar.radio(
        "Select view", ["Single-turn (simple+advanced)", "Longitudinal"], index=0
    )

    if view.startswith("Single-turn"):
        render_single_turn(single_data or [])
    else:
        render_longitudinal(long_data or {})


if __name__ == "__main__":
    main()
