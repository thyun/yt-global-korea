"""
topic_agent.py — Transcript 분석 기반 에피소드 주제 선정 에이전트 (파이프라인 2단계)
"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from .base_agent import BaseAgent

console = Console()


class TopicAgent(BaseAgent):
    """
    Research Agent가 수집한 외국어 transcript를 분석하여
    새 에피소드 주제를 선정합니다.
    """

    def run(self) -> dict:
        console.rule("[bold cyan]2단계: Transcript 분석 기반 주제 선정 (Topic Agent)")

        research_data = self.read_json("01_research/videos.json")
        transcripts = self._load_transcripts()
        past_topics = self._get_past_topics()

        if transcripts:
            console.print(f"  [green]{len(transcripts)}개 transcript 로드 완료[/green]")
            topic_data = self._select_topic_from_transcripts(research_data, transcripts, past_topics)
        else:
            console.print("  [yellow]transcript 없음 → 랭킹 데이터만으로 주제 선정[/yellow]")
            topic_data = self._select_topic_from_ranking(research_data, past_topics)

        self.write_json("02_topic/topic.json", topic_data)
        console.print(f"  [bold]선정된 주제:[/bold] {topic_data.get('title', '')}")
        console.print(f"  [dim]{topic_data.get('subtitle', '')}[/dim]")
        return topic_data

    # ------------------------------------------------------------------ #
    # Transcript 로드                                                        #
    # ------------------------------------------------------------------ #
    def _load_transcripts(self) -> list[dict]:
        """다운로드된 transcript 파일들을 로드하여 반환"""
        transcript_dir = self.episode_dir / "01_research" / "transcripts"
        if not transcript_dir.exists():
            return []

        results = []
        research_data = self.read_json("01_research/videos.json")
        video_map = {v["video_id"]: v for v in research_data.get("videos", [])}

        for txt_file in sorted(transcript_dir.glob("*_transcript.txt")):
            vid_id = txt_file.stem.replace("_transcript", "")
            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            video_info = video_map.get(vid_id, {})
            results.append({
                "video_id": vid_id,
                "title": video_info.get("title", ""),
                "channel": video_info.get("channel", ""),
                "views": video_info.get("views", 0),
                "rank": video_info.get("rank", 99),
                "transcript": text[:3000],  # LLM 입력 제한을 위해 앞 3000자만
                "transcript_length": len(text),
            })

        # 랭킹 순으로 정렬, 최대 2개만 사용 (원본 영상 참조 제한)
        results.sort(key=lambda x: x["rank"])
        return results[:2]

    # ------------------------------------------------------------------ #
    # Transcript 기반 주제 선정                                              #
    # ------------------------------------------------------------------ #
    def _select_topic_from_transcripts(
        self, research_data: dict, transcripts: list[dict], past_topics: list[str]
    ) -> dict:
        system_prompt = self.load_template("topic_prompt.txt")

        # 각 transcript 요약 블록 구성
        transcript_blocks = []
        for t in transcripts:
            block = f"""### [{t['rank']}위] {t['title']} — {t['channel']} ({t['views']:,}회)
{t['transcript'][:1500]}
---"""
            transcript_blocks.append(block)

        past_str = "\n".join(f"- {p}" for p in past_topics) if past_topics else "없음"
        trend_analysis = research_data.get("trend_analysis", "")

        user_prompt = f"""## 외국어 YouTube 영상 Transcript 분석 자료

아래는 외국인이 직접 제작한 '한국 생활' 관련 인기 영상의 transcript입니다.
(한국어 영상 제외, 영어/외국어 영상만 포함)

{chr(10).join(transcript_blocks)}

## 트렌드 분석
{trend_analysis}

## 이미 제작한 에피소드 (중복 금지)
{past_str}

## 분석 요청
위 transcript들에서:
1. 외국인들이 공통적으로 언급하는 주제나 감정은 무엇인가요?
2. 어떤 주제가 시청자 반응(댓글, 공감)을 많이 이끌어내고 있나요?
3. 아직 충분히 다뤄지지 않은 틈새 주제가 있나요?

이 분석을 바탕으로 새 에피소드 주제를 선정해 주세요.
반드시 JSON 형식으로만 응답하세요."""

        raw = self.call_llm(system_prompt, user_prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    # 랭킹 기반 주제 선정 (transcript 없을 때 fallback)                       #
    # ------------------------------------------------------------------ #
    def _select_topic_from_ranking(self, research_data: dict, past_topics: list[str]) -> dict:
        videos = research_data.get("videos", [])
        trend_analysis = research_data.get("trend_analysis", "")
        top_videos_str = "\n".join(
            f"{v.get('rank', i+1)}위. [{v.get('views', 0):,}회] {v.get('title', '')} — {v.get('channel', '')}"
            for i, v in enumerate(videos[:15])
        )
        past_str = "\n".join(f"- {p}" for p in past_topics) if past_topics else "없음"

        system_prompt = self.load_template("topic_prompt.txt")
        user_prompt = f"""## 외국어 인기 동영상 랭킹 (한국어 제외)
{top_videos_str}

## 트렌드 분석
{trend_analysis}

## 이미 제작한 에피소드 (중복 금지)
{past_str}

위 정보를 바탕으로 새 에피소드 주제를 선정해 주세요.
반드시 JSON 형식으로만 응답하세요."""

        raw = self.call_llm(system_prompt, user_prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    # 이전 에피소드 주제 목록                                                 #
    # ------------------------------------------------------------------ #
    def _get_past_topics(self) -> list[str]:
        episodes_dir = Path(self.settings["paths"]["episodes_dir"])
        topics: list[str] = []
        if not episodes_dir.exists():
            return topics
        for ep_dir in sorted(episodes_dir.iterdir()):
            if ep_dir == self.episode_dir:
                continue
            topic_file = ep_dir / "02_topic" / "topic.json"
            if topic_file.exists():
                try:
                    data = json.loads(topic_file.read_text(encoding="utf-8"))
                    topics.append(data.get("title", ""))
                except Exception:
                    pass
        return topics

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)


