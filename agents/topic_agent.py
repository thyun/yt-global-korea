"""
topic_agent.py — 리서치 결과 기반 에피소드 주제 선정 에이전트 (파이프라인 2단계)
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class TopicAgent(BaseAgent):
    """
    Research Agent가 찾은 인기 동영상 목록을 분석하여
    새 에피소드 주제를 선정합니다.
    """

    def run(self) -> dict:
        console.rule("[bold cyan]2단계: 주제 선정 (Topic Agent)")

        research_data = self.read_json("01_research/videos.json")
        past_topics = self._get_past_topics()

        system_prompt = self.load_template("topic_prompt.txt")
        user_prompt = self._build_user_prompt(research_data, past_topics)

        raw = self.call_llm(system_prompt, user_prompt)
        topic_data = self._parse_json(raw)
        self.write_json("02_topic/topic.json", topic_data)

        console.print(f"  [bold]선정된 주제:[/bold] {topic_data.get('title', '')}")
        console.print(f"  [dim]{topic_data.get('subtitle', '')}[/dim]")
        return topic_data

    def _get_past_topics(self) -> list[str]:
        episodes_dir = Path(self.settings["paths"]["episodes_dir"])
        topics: list[str] = []
        if not episodes_dir.exists():
            return topics
        for ep_dir in sorted(episodes_dir.iterdir()):
            topic_file = ep_dir / "02_topic" / "topic.json"
            if topic_file.exists():
                try:
                    data = json.loads(topic_file.read_text(encoding="utf-8"))
                    topics.append(data.get("title", ""))
                except Exception:
                    pass
        return topics

    def _build_user_prompt(self, research_data: dict, past_topics: list[str]) -> str:
        videos = research_data.get("videos", [])
        trend_analysis = research_data.get("trend_analysis", "")

        top_videos_str = "\n".join(
            f"{v.get('rank', i+1)}위. [{v.get('views', 0):,}회] {v.get('title', '')} — {v.get('channel', '')}"
            for i, v in enumerate(videos[:20])
        )
        past_str = "\n".join(f"- {t}" for t in past_topics) if past_topics else "없음"

        return f"""## 인기 동영상 랭킹 (Research Agent 결과)
{top_videos_str}

## 트렌드 분석
{trend_analysis}

## 이미 다룬 주제 (중복 금지)
{past_str}

위 정보를 바탕으로 새 에피소드 주제를 선정해 주세요.
반드시 JSON 형식으로만 응답하세요."""

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

