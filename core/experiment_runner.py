from typing import Any, Dict, List
import logging

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
            reflection.get("revision_instruction", "")
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

        for model_cfg in self.client_model_configs:
            logger.info("Running single-turn tasks for model: %s", model_cfg["id"])

            for task in self.tasks:
                baseline = self.run_baseline(model_cfg, task)
                cbt_output = self.run_cbt(model_cfg, task)

                all_results.append(
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

        return all_results
