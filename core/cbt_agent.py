import json
import logging
from typing import Any, Dict

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


CBT_DETECTION_PROMPT = """
You are an AI-CBT meta-agent evaluating another AI model's reasoning.

Your goal is to detect cognitive-style distortions and guide the model
toward clearer, more grounded reasoning.

Analyze the following model output:

--- BEGIN OUTPUT ---
{agent_output}
--- END OUTPUT ---

Identify any distortions such as:
- overgeneralization
- black-and-white (all-or-nothing) thinking
- unwarranted certainty or overconfidence
- hallucinated justifications or facts
- ignoring important counterexamples or caveats
- contradictions within the answer
- oversimplified or shallow explanations
- emotionally loaded language inappropriate for an AI

Then:

1. List each distortion type you detect.
2. Briefly explain how they show up in this output.
3. Provide high-level corrective guidance.
4. Produce a concrete "revision instruction" that the original model
   can follow to generate a better answer.

Return your response in pure JSON, with this exact structure:

{{
  "distortions": ["..."],
  "explanation": "...",
  "guidance": "...",
  "revision_instruction": "..."
}}
"""


CBT_REVISION_WRAPPER = """
You produced a previous answer that contained some reasoning issues.

Here is your previous answer:
---
{previous}
---

Follow this revision instruction:
"{instruction}"

Revise your prior answer to:
- remove or reduce the reasoning distortions,
- add nuance and grounded reasoning,
- avoid hallucinations and unjustified certainty,
- remain clear and logically structured.

Now produce your corrected final answer.
"""


class CBTAgent:
    """
    CBT meta-agent that:
    - inspects an LLM's output,
    - identifies distortions,
    - produces a revision instruction,
    - helps construct a revision prompt.
    """

    def __init__(self, model_name: str) -> None:
        self.client = LLMClient(model_name=model_name, temperature=0.3, max_tokens=1024)

    def evaluate(self, agent_output: str) -> Dict[str, Any]:
        prompt = CBT_DETECTION_PROMPT.format(agent_output=agent_output)
        logger.info("Running CBT evaluation on agent output")
        raw = self.client.complete(prompt)

        # best-effort JSON parsing with fallback
        try:
            parsed = json.loads(raw)
        except Exception:
            logger.warning("CBT agent returned non-JSON, attempting repair")
            repair_prompt = f"""
Convert the following text into valid JSON without changing the semantic content.
If it already looks like JSON but has minor issues, fix them.

TEXT:
{raw}
"""
            repaired = self.client.complete(repair_prompt)
            parsed = json.loads(repaired)

        # ensure keys exist with defaults
        if "revision_instruction" not in parsed:
            parsed["revision_instruction"] = (
                "Re-evaluate your answer and improve clarity, nuance, and grounding, "
                "reducing any distortions or overconfidence."
            )
        if "distortions" not in parsed:
            parsed["distortions"] = []
        if "explanation" not in parsed:
            parsed["explanation"] = ""
        if "guidance" not in parsed:
            parsed["guidance"] = ""

        return parsed

    def build_revision_prompt(
        self, revision_instruction: str, previous_answer: str
    ) -> str:
        return CBT_REVISION_WRAPPER.format(
            instruction=revision_instruction, previous=previous_answer
        )
