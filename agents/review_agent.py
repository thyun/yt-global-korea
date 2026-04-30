"""
review_agent.py — 스크립트 검토 및 개선 에이전트
"""
from __future__ import annotations

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class ReviewAgent(BaseAgent):
    """작성된 스크립트를 검토하고 품질을 개선합니다."""

    def run(self) -> str:
        console.rule("[bold cyan]4단계: 스크립트 검토 (Review Agent)")

        topic_data = self.read_json("02_topic/topic.json")
        script_md = self.read_file("03_script/script.md")
        target_words = self.settings["channel"]["target_word_count"]

        system_prompt = self.load_template("review_prompt.txt")
        user_prompt = self._build_user_prompt(topic_data, script_md, target_words)

        revised_md = self.call_llm(system_prompt, user_prompt)
        self.write_file("04_review/script_revised.md", revised_md)
        return revised_md

    def _build_user_prompt(self, topic_data: dict, script_md: str, target_words: int) -> str:
        return f"""## 에피소드 주제
{topic_data.get('title', '')} — {topic_data.get('subtitle', '')}

## 검토 기준
- 목표 분량: 약 {target_words}자
- 자연스러운 한국어 구어체
- 유튜브 시청자 집중도 유지 (훅, 전환, 마무리)
- 정보의 정확성과 흥미로운 전달

## 원본 스크립트
{script_md}

위 스크립트를 검토하고 개선된 최종 버전을 작성해 주세요.
마지막에 ## 변경 사항 섹션을 추가하여 수정 내용을 간략히 정리해 주세요."""
