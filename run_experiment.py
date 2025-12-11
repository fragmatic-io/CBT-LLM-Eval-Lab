import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from core.task_loader import load_tasks
from core.experiment_runner import ExperimentRunner
from core.evaluator import Evaluator


def setup_logging(output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"experiment_{timestamp}.log"
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

    simple_tasks = load_tasks(str(config_dir / "tasks_simple.yaml")).get(
        "simple_tasks", []
    )
    advanced_tasks = load_tasks(str(config_dir / "tasks_advanced.yaml")).get(
        "advanced_tasks", []
    )

    all_tasks: List[Dict[str, Any]] = simple_tasks + advanced_tasks

    runner = ExperimentRunner(
        client_model_configs=config["client_models"],
        cbt_model_config=config["cbt_model"],
        tasks=all_tasks,
    )

    logging.info(
        "Starting single-turn run with %s client models across %s tasks",
        len(config["client_models"]),
        len(all_tasks),
    )
    results = runner.run_all()

    evaluator = Evaluator(config["evaluator_models"])

    # attach evaluator scores for each (model, task)
    for rec in results:
        baseline_text = rec["condition_results"]["baseline"]
        cbt_text = rec["condition_results"]["cbt"]["revised"]
        try:
            scores = evaluator.score_pair(baseline_text, cbt_text)
        except Exception as exc:  # noqa: BLE001
            logging.error(
                "Evaluator scoring failed for model=%s task=%s: %s",
                rec["model_id"],
                rec["task_id"],
                exc,
            )
            scores = []
        rec["evaluation"] = scores

    out_path = output_dir / "single_turn_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info("Single-turn experiment complete → %s", out_path)


if __name__ == "__main__":
    main()
