import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from core.task_loader import load_tasks
from core.longitudinal_runner import LongitudinalRunner


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

    all_results: Dict[str, Any] = {}

    for model_cfg in config["client_models"]:
        logging.info("Running longitudinal tasks for model: %s", model_cfg["id"])
        model_record: Dict[str, Any] = {}

        for task in tasks_long:
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

            model_record[task["id"]] = {
                "task_prompt": task["prompt"],
                "baseline_history": baseline_hist,
                "cbt_history": cbt_hist,
            }

        all_results[model_cfg["id"]] = {
            "model_name": model_cfg["name"],
            "tasks": model_record,
        }

    out_path = output_dir / "longitudinal_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logging.info("Longitudinal experiment complete → %s", out_path)


if __name__ == "__main__":
    main()
