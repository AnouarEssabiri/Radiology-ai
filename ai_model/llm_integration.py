"""
ai_model/llm_integration.py
============================
Integration with LLMs (e.g., Groq) for high-quality radiology report generation.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LLMReportGenerator:
    """
    LLM-based report generator using Groq API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.system_prompt = system_prompt or "You are a highly skilled radiologist."
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            logger.warning("No GROQ_API_KEY provided. LLM integration will be disabled.")
            self.client = None
        else:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"LLM client initialized with model: {self.model}")
            except ImportError:
                logger.error("Groq library not installed. Please install it with 'pip install groq'")
                self.client = None

    def generate_report(self, base_report: Optional[str] = None) -> str:
        """
        Generate a high-quality radiology report.

        Args:
            base_report: Optional base report from the vision model to refine.

        Returns:
            Generated report string.
        """
        if self.client is None:
            logger.warning("LLM client not available. Returning base report.")
            return base_report or "No report available."

        try:
            user_prompt = "Generate a professional radiology report for a chest X-ray."
            if base_report:
                user_prompt = f"Refine and improve this radiology report to make it more professional and clinically accurate:\n\n{base_report}"

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            report = chat_completion.choices[0].message.content.strip()
            logger.info("LLM report generated successfully.")
            return report

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return base_report or "Failed to generate report."

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMReportGenerator":
        """
        Create an LLMReportGenerator from config dict.
        """
        llm_cfg = cfg.get("llm", {})
        return cls(
            model=llm_cfg.get("model", "llama-3.3-70b-versatile"),
            api_key=llm_cfg.get("api_key"),
            system_prompt=llm_cfg.get("system_prompt"),
            max_tokens=llm_cfg.get("max_tokens", 500),
            temperature=llm_cfg.get("temperature", 0.7),
        )
