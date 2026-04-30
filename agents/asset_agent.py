"""
asset_agent.py — 씬별 에셋 구성 에이전트
"""
from __future__ import annotations

import json

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class AssetAgent(BaseAgent):
    """수정된 스크립트를 바탕으로 씬별 이미지/영상 프롬프트를 생성합니다."""

    def run(self) -> list:
        console.rule("[bold cyan]5단계: 에셋 구성 (Asset Agent)")

        topic_data = self.read_json("02_topic/topic.json")
        script_md = self.read_file("04_review/script_revised.md")

        system_prompt = self.load_template("asset_prompt.txt")
        user_prompt = self._build_user_prompt(topic_data, script_md)

        raw = self.call_llm(system_prompt, user_prompt)
        assets = self._parse_json(raw)
        self.write_json("05_assets/assets.json", assets)
        return assets

    def _build_user_prompt(self, topic_data: dict, script_md: str) -> str:
        return f"""## 에피소드 주제
{topic_data.get('title', '')}

## 최종 스크립트
{script_md}

위 스크립트를 씬 단위로 분석하여 각 씬에 필요한 영상/이미지 에셋 정보를 JSON으로 작성해 주세요.
반드시 JSON 배열 형식으로만 응답하세요."""

    @staticmethod
    def _parse_json(raw: str) -> list:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)
