"""
production_agent.py — 최종 제작 가이드 에이전트
"""
from __future__ import annotations

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class ProductionAgent(BaseAgent):
    """모든 산출물을 종합하여 최종 제작 가이드를 생성합니다."""

    def run(self) -> str:
        console.rule("[bold cyan]6단계: 제작 가이드 (Production Agent)")

        topic_data = self.read_json("02_topic/topic.json")
        script_md = self.read_file("04_review/script_revised.md")
        assets = self.read_json("05_assets/assets.json")

        system_prompt = self.load_template("production_prompt.txt")
        user_prompt = self._build_user_prompt(topic_data, script_md, assets)

        guide_md = self.call_llm(system_prompt, user_prompt)
        self.write_file("06_production/production_guide.md", guide_md)
        return guide_md

    def _build_user_prompt(self, topic_data: dict, script_md: str, assets: list) -> str:
        import json
        assets_summary = json.dumps(assets[:5], ensure_ascii=False, indent=2)
        return f"""## 에피소드 정보
제목: {topic_data.get('title', '')}
부제: {topic_data.get('subtitle', '')}
키워드: {', '.join(topic_data.get('keywords', []))}

## 최종 스크립트 (요약)
{script_md[:800]}...

## 에셋 목록 (앞부분 미리보기)
{assets_summary}

위 정보를 바탕으로 동영상 편집자를 위한 최종 제작 가이드를 작성해 주세요."""
