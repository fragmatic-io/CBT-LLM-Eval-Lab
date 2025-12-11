import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from core.task_loader import load_tasks
from core.longitudinal_runner import LongitudinalRunner
from core.evaluator import Evaluator


def setup_logging(output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"longitudinal_{timestamp}.log"
    handlers = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.info("Logging initialized. Writing to %s", log_path)


def main() -> None:
    load_dotenv()
    base_dir = Path(__file__).resolve().parent
    config_dir = base_dir / "config"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    with open(config_dir / "models.yaml", "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    tasks_long = load_tasks(str(config_dir / "tasks_longitudinal.yaml")).get(
        "longitudinal_tasks", []
    )

    rounds = 5  # you can tune this
    runner = LongitudinalRunner(
        rounds=rounds,
        cbt_model_config=config["cbt_model"],
    )
    evaluator = Evaluator(config["evaluator_models"])

    all_results: Dict[str, Any] = {}

    def run_single_task(model_cfg: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        logging.info(
            "Running longitudinal task concurrently: model=%s task=%s",
            model_cfg["id"],
            task["id"],
        )
        baseline_hist = runner.run_condition(
            model_cfg=model_cfg,
            task=task,
            condition="baseline",
        )
        cbt_hist = runner.run_condition(
            model_cfg=model_cfg,
            task=task,
            condition="cbt",
        )
        baseline_final = baseline_hist[-1].get("raw", "")
        cbt_final = cbt_hist[-1].get("revised") or cbt_hist[-1].get("raw", "")
        eval_scores = evaluator.score_pair(baseline_final, cbt_final)
        return {
            "model_id": model_cfg["id"],
            "model_name": model_cfg["name"],
            "task_id": task["id"],
            "task_prompt": task["prompt"],
            "baseline_history": baseline_hist,
            "cbt_history": cbt_hist,
            "baseline_final": baseline_final,
            "cbt_final": cbt_final,
            "final_evaluation": eval_scores,
        }

    # run each model/task pair concurrently (up to 32 threads)
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_key = {}
        for model_cfg in config["client_models"]:
            for task in tasks_long:
                fut = executor.submit(run_single_task, model_cfg, task)
                future_to_key[fut] = (model_cfg["id"], task["id"])

        total = len(future_to_key)
        completed = 0
        for fut in as_completed(future_to_key):
            model_id, task_id = future_to_key[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                logging.error("Longitudinal run failed for model=%s task=%s: %s", model_id, task_id, exc)
                continue

            if model_id not in all_results:
                all_results[model_id] = {"model_name": res["model_name"], "tasks": {}}
            all_results[model_id]["tasks"][task_id] = {
                "task_prompt": res["task_prompt"],
                "baseline_history": res["baseline_history"],
                "cbt_history": res["cbt_history"],
                "baseline_final": res["baseline_final"],
                "cbt_final": res["cbt_final"],
                "final_evaluation": res["final_evaluation"],
            }
            completed += 1
            logging.info(
                "Longitudinal progress: %s/%s (%.0f%%)",
                completed,
                total,
                (completed / total) * 100,
            )

    out_path = output_dir / "longitudinal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logging.info("Longitudinal experiment complete → %s", out_path)


if __name__ == "__main__":
    main()
