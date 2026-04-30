"""
research_agent.py — YouTube 동영상 검색 및 랭킹 에이전트 (파이프라인 1단계)

YouTube Data API v3가 설정된 경우 실제 검색 수행,
YOUTUBE_API_KEY 미설정 시 LLM 기반 시뮬레이션으로 대체.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from rich.console import Console
from rich.table import Table

from .base_agent import BaseAgent

console = Console()


class ResearchAgent(BaseAgent):
    """
    외국인 한국 생활 관련 YouTube 동영상을 검색하고 랭킹을 매깁니다.

    출력: 01_research/videos.json
    """

    def run(self) -> dict:
        console.rule("[bold cyan]1단계: 소재 리서치 (Research Agent)")

        api_key = os.getenv("YOUTUBE_API_KEY", "")
        if api_key and not api_key.startswith("AIza-your"):
            console.print("  [green]YouTube Data API v3 사용[/green]")
            result = self._search_youtube(api_key)
        else:
            console.print("  [yellow]YOUTUBE_API_KEY 미설정 → LLM 시뮬레이션 모드[/yellow]")
            result = self._simulate_with_llm()

        self.write_json("01_research/videos.json", result)
        self._print_ranking_table(result["videos"])
        return result

    # ------------------------------------------------------------------ #
    # YouTube Data API v3 실제 검색                                        #
    # ------------------------------------------------------------------ #
    def _search_youtube(self, api_key: str) -> dict:
        from googleapiclient.discovery import build

        queries = self.settings["research"]["youtube_search_queries"]
        max_per_query = self.settings["research"]["max_results_per_query"]
        top_n = self.settings["research"]["top_videos_to_rank"]

        youtube = build("youtube", "v3", developerKey=api_key)
        seen_ids: set[str] = set()
        raw_videos: list[dict] = []

        for query in queries:
            console.print(f"  [dim]검색: {query}[/dim]")
            search_resp = youtube.search().list(
                q=query,
                part="snippet",
                type="video",
                maxResults=max_per_query,
                relevanceLanguage="ko",
                regionCode="KR",
            ).execute()

            video_ids = [
                item["id"]["videoId"]
                for item in search_resp.get("items", [])
                if item["id"]["videoId"] not in seen_ids
            ]
            seen_ids.update(video_ids)

            if not video_ids:
                continue

            # 통계 정보 가져오기
            stats_resp = youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=",".join(video_ids),
            ).execute()

            for item in stats_resp.get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                raw_videos.append({
                    "video_id": item["id"],
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", "")[:300],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "search_query": query,
                })

        ranked = self._rank_videos(raw_videos, top_n)
        trend_analysis = self._analyze_trends(ranked)
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "search_queries": queries,
            "total_found": len(raw_videos),
            "videos": ranked,
            "trend_analysis": trend_analysis,
        }

    # ------------------------------------------------------------------ #
    # LLM 시뮬레이션 (API 키 없을 때)                                        #
    # ------------------------------------------------------------------ #
    def _simulate_with_llm(self) -> dict:
        queries = self.settings["research"]["youtube_search_queries"]
        top_n = self.settings["research"]["top_videos_to_rank"]

        system_prompt = self.load_template("research_prompt.txt")
        user_prompt = f"""다음 검색 키워드로 YouTube에서 외국인 한국 생활 관련 동영상을 검색한다고 가정하고,
실제로 존재할 법한 인기 동영상 {top_n}개의 목록을 만들어 주세요.

검색 키워드:
{chr(10).join(f'- {q}' for q in queries)}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "searched_at": "ISO8601 날짜",
  "search_queries": ["키워드1", ...],
  "total_found": {top_n},
  "videos": [
    {{
      "rank": 1,
      "video_id": "가상 ID",
      "title": "동영상 제목",
      "channel": "채널명",
      "published_at": "YYYY-MM-DDTHH:MM:SSZ",
      "description": "동영상 설명 (200자 이내)",
      "url": "https://www.youtube.com/watch?v=가상ID",
      "views": 조회수(정수),
      "likes": 좋아요수(정수),
      "comments": 댓글수(정수),
      "search_query": "해당 검색 키워드",
      "main_topics": ["주제1", "주제2"],
      "ranking_score": 0.0~1.0 사이 점수,
      "ranking_reason": "랭킹 이유 한 줄"
    }}
  ],
  "trend_analysis": "전반적인 트렌드 분석 (300자 이내)"
}}"""

        raw = self.call_llm(system_prompt, user_prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    # 랭킹 계산                                                             #
    # ------------------------------------------------------------------ #
    def _rank_videos(self, videos: list[dict], top_n: int) -> list[dict]:
        weights = self.settings["research"]["ranking_weights"]
        now = datetime.now(timezone.utc)

        # 최대값 정규화를 위한 기준값
        max_views = max((v["views"] for v in videos), default=1) or 1
        max_likes = max((v["likes"] for v in videos), default=1) or 1

        for v in videos:
            # 최신성 점수: 최근 2년 이내면 1.0, 오래될수록 감소
            try:
                pub = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
                days_old = (now - pub).days
                recency = max(0.0, 1.0 - days_old / 730)
            except Exception:
                recency = 0.5

            score = (
                weights["views"] * math.log1p(v["views"]) / math.log1p(max_views)
                + weights["likes"] * math.log1p(v["likes"]) / math.log1p(max_likes)
                + weights["recency"] * recency
                + weights["relevance"] * 0.8  # 검색 결과이므로 기본 관련성 부여
            )
            v["ranking_score"] = round(score, 4)

        ranked = sorted(videos, key=lambda x: x["ranking_score"], reverse=True)[:top_n]
        for i, v in enumerate(ranked, 1):
            v["rank"] = i

        trend_analysis = self._analyze_trends(ranked)
        return ranked

    def _analyze_trends(self, ranked_videos: list[dict]) -> str:
        if not ranked_videos:
            return ""
        titles = [v["title"] for v in ranked_videos[:10]]
        system_prompt = "당신은 유튜브 콘텐츠 트렌드 분석 전문가입니다."
        user_prompt = f"""다음은 '외국인 한국 생활' 관련 인기 YouTube 동영상 상위 10개 제목입니다.
이 목록에서 보이는 콘텐츠 트렌드, 인기 주제 패턴, 시청자 관심사를 300자 이내로 분석해 주세요.

{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(titles))}"""
        return self.call_llm(system_prompt, user_prompt)

    # ------------------------------------------------------------------ #
    # 결과 출력                                                             #
    # ------------------------------------------------------------------ #
    def _print_ranking_table(self, videos: list[dict]) -> None:
        table = Table(title="📊 외국인 한국 생활 인기 동영상 랭킹", show_lines=True)
        table.add_column("순위", style="bold yellow", width=4)
        table.add_column("제목", style="cyan", max_width=40)
        table.add_column("채널", style="green", max_width=20)
        table.add_column("조회수", justify="right", width=10)
        table.add_column("점수", justify="right", width=6)

        for v in videos[:15]:
            table.add_row(
                str(v.get("rank", "?")),
                v.get("title", "")[:40],
                v.get("channel", "")[:20],
                f"{v.get('views', 0):,}",
                str(v.get("ranking_score", ""))[:5],
            )
        console.print(table)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)

