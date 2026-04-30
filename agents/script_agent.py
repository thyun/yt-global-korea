"""
script_agent.py — 스크립트 작성 에이전트
"""
from __future__ import annotations

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class ScriptAgent(BaseAgent):
    """리서치 자료를 바탕으로 8분 분량의 한국어 스크립트를 작성합니다."""

    def run(self) -> str:
        console.rule("[bold cyan]3단계: 스크립트 작성 (Script Agent)")

        topic_data = self.read_json("02_topic/topic.json")
        research_data = self.read_json("01_research/videos.json")
        target_words = self.settings["channel"]["target_word_count"]

        system_prompt = self.load_template("script_prompt.txt")
        user_prompt = self._build_user_prompt(topic_data, research_data, target_words)

        script_md = self.call_llm(system_prompt, user_prompt)
        self.write_file("03_script/script.md", script_md)
        return script_md

    def _build_user_prompt(self, topic_data: dict, research_data: dict, target_words: int) -> str:
        videos = research_data.get("videos", [])
        trend_analysis = research_data.get("trend_analysis", "")
        top_videos_str = "\n".join(
            f"- [{v.get('views', 0):,}회] {v.get('title', '')} ({v.get('channel', '')})"
            for v in videos[:10]
        )
        # transcript가 있는 영상만 원본영상 삽입 후보로 제공
        transcript_videos = [v for v in videos if v.get("has_transcript")]
        transcript_ref_str = ""
        if transcript_videos:
            lines = []
            for v in transcript_videos[:5]:
                lines.append(
                    f'- 제목: "{v.get("title", "")}" | 채널: {v.get("channel", "")} | '
                    f'video_id: {v.get("video_id", "")} | 조회수: {v.get("views", 0):,}회'
                )
            transcript_ref_str = (
                "\n## 원본영상 삽입 후보 (transcript 확보된 영상)\n"
                "아래 영상들의 클립을 스크립트에서 [원본영상 삽입] 마커로 삽입할 수 있습니다.\n"
                "삽입 시 MM:SS~MM:SS 형식으로 가장 핵심 발언 구간을 지정하세요 (최대 5회 이내).\n\n"
                + "\n".join(lines)
            )

        return f"""## 에피소드 정보
주제: {topic_data.get('title', '')}
부제: {topic_data.get('subtitle', '')}
선정 이유: {topic_data.get('selection_reason', '')}
핵심 각도: {topic_data.get('angle', '')}
오프닝 훅: {topic_data.get('hook', '')}
목표 분량: 약 {target_words}자 (한국어 기준 8분 낭독)
{transcript_ref_str}
## 참고 인기 동영상 (소재 참고용)
{top_videos_str}

## 트렌드 분석
{trend_analysis}

위 정보를 바탕으로 유튜브 동영상 스크립트를 작성해 주세요.
반드시 [나레이션], [B-ROLL: ...], [원본영상 삽입: ...] 블록 포맷을 준수하세요."""
