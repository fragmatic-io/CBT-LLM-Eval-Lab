from typing import Any, Dict, List, Literal
import logging

from core.llm_client import LLMClient
from core.cbt_agent import CBTAgent

logger = logging.getLogger(__name__)


Condition = Literal["baseline", "cbt"]


class LongitudinalRunner:
    """
    Runs multi-round tasks under two conditions:
    - baseline: no CBT meta feedback
    - cbt: each round uses CBT reflection + revision
    """

    def __init__(
        self,
        rounds: int,
        cbt_model_config: Dict[str, Any],
    ) -> None:
        self.rounds = rounds
        self.cbt_model_config = cbt_model_config

    def _build_client(self, model_cfg: Dict[str, Any]) -> LLMClient:
        return LLMClient(
            model_name=model_cfg["name"],
            max_tokens=model_cfg.get("max_tokens", 2048),
            temperature=model_cfg.get("temperature", 0.7),
        )

    def run_condition(
        self,
        model_cfg: Dict[str, Any],
        task: Dict[str, Any],
        condition: Condition,
    ) -> List[Dict[str, Any]]:
        """
        Returns per-round history for one model, one task, and one condition.
        """
        llm = self._build_client(model_cfg)
        history: List[Dict[str, Any]] = []
        last_output: str | None = None

        cbt_agent = None
        if condition == "cbt":
            cbt_agent = CBTAgent(self.cbt_model_config["name"])

        for r in range(1, self.rounds + 1):
            logger.info(
                "Round %s/%s: model=%s task=%s condition=%s",
                r,
                self.rounds,
                model_cfg["id"],
                task["id"],
                condition,
            )
            if r == 1:
                prompt = task["prompt"]
            else:
                template = task["round_prompt_template"]
                prompt = template.format(previous=last_output)

            raw = llm.complete(prompt)

            if condition == "baseline":
                # no CBT feedback, raw is the round output
                round_record = {
                    "round": r,
                    "prompt": prompt,
                    "raw": raw,
                }
                last_output = raw
            else:
                assert cbt_agent is not None
                reflection = cbt_agent.evaluate(raw)
                revision_prompt = cbt_agent.build_revision_prompt(
                    revision_instruction=reflection.get("revision_instruction", ""),
                    previous_answer=raw,
                )
                revised = llm.complete(revision_prompt)

                round_record = {
                    "round": r,
                    "prompt": prompt,
                    "raw": raw,
                    "reflection": reflection,
                    "revision_prompt": revision_prompt,
                    "revised": revised,
                }
                last_output = revised

            history.append(round_record)

        return history
