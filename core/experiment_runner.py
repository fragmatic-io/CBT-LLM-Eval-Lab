from typing import Any, Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.llm_client import LLMClient
from core.cbt_agent import CBTAgent

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs baseline vs CBT-assisted conditions across tasks and client models
    for single-turn tasks (simple + advanced).
    """

    def __init__(
        self,
        client_model_configs: List[Dict[str, Any]],
        cbt_model_config: Dict[str, Any],
        tasks: List[Dict[str, Any]],
    ) -> None:
        self.client_model_configs = client_model_configs
        self.cbt_model_config = cbt_model_config
        self.tasks = tasks

    def _build_client(self, model_cfg: Dict[str, Any]) -> LLMClient:
        return LLMClient(
            model_name=model_cfg["name"],
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=model_cfg.get("temperature", 0.7),
        )

    def run_baseline(self, model_cfg: Dict[str, Any], task: Dict[str, Any]) -> str:
        llm = self._build_client(model_cfg)
        logger.info("Running baseline: model=%s task=%s", model_cfg["id"], task["id"])
        return llm.complete(task["prompt"])

    def run_cbt(self, model_cfg: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        llm = self._build_client(model_cfg)
        cbt = CBTAgent(self.cbt_model_config["name"])

        raw = llm.complete(task["prompt"])
        reflection = cbt.evaluate(raw)
        revision_prompt = cbt.build_revision_prompt(
            revision_instruction=reflection.get("revision_instruction", ""),
            previous_answer=raw,
        )
        revised = llm.complete(revision_prompt)

        logger.info(
            "Completed CBT pass: model=%s task=%s distortions=%s",
            model_cfg["id"],
            task["id"],
            reflection.get("distortions"),
        )

        return {
            "raw": raw,
            "reflection": reflection,
            "revised": revised,
        }

    def run_all(self) -> List[Dict[str, Any]]:
        """
        Returns a list of records, one per model x task, each containing:
        - baseline output
        - CBT condition outputs (raw, reflection, revised)
        """
        all_results: List[Dict[str, Any]] = []
        total_tasks = len(self.tasks) * len(self.client_model_configs)
        completed_tasks = 0
        progress_lock = threading.Lock()

        def run_for_model(model_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
            nonlocal completed_tasks
            logger.info("Running single-turn tasks for model: %s", model_cfg["id"])
            model_results: List[Dict[str, Any]] = []

            for task in self.tasks:
                baseline = self.run_baseline(model_cfg, task)
                cbt_output = self.run_cbt(model_cfg, task)

                model_results.append(
                    {
                        "model_id": model_cfg["id"],
                        "model_name": model_cfg["name"],
                        "task_id": task["id"],
                        "task_prompt": task["prompt"],
                        "condition_results": {
                            "baseline": baseline,
                            "cbt": cbt_output,
                        },
                    }
                )

                with progress_lock:
                    nonlocal completed_tasks
                    completed_tasks += 1
                    pct = (completed_tasks / total_tasks) * 100
                    logger.info(
                        "Task progress: %.1f%% (%d/%d) model=%s task=%s",
                        pct,
                        completed_tasks,
                        total_tasks,
                        model_cfg["id"],
                        task["id"],
                    )

            return model_results

        with ThreadPoolExecutor(max_workers=len(self.client_model_configs)) as executor:
            future_to_model = {
                executor.submit(run_for_model, cfg): cfg for cfg in self.client_model_configs
            }
            for fut in as_completed(future_to_model):
                results = fut.result()
                all_results.extend(results)

        return all_results
