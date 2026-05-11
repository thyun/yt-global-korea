"""
research_agent.py — 외국어 YouTube 동영상 검색 + Transcript 저장 에이전트 (파이프라인 1단계)

- 한국어 영상 제외, 영어/외국어 영상만 검색
- youtube-transcript-api로 transcript 다운로드
- 01_research/videos.json + 01_research/transcripts/ 에 저장
- YouTube Data API v3 없으면 LLM 시뮬레이션 모드
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .base_agent import BaseAgent

console = Console()

# 한국어 채널/제목 판별용 패턴
_KOREAN_RE = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")

# ISO 8601 duration 파싱 (예: PT5M30S → 330)
_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_duration_seconds(iso_duration: str) -> int:
    """ISO 8601 duration 문자열을 초로 변환. 파싱 실패 시 0 반환."""
    m = _DURATION_RE.match(iso_duration or "")
    if not m:
        return 0
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _is_korean_content(title: str, channel: str, language_code: str | None) -> bool:
    """제목·채널명·언어 코드를 기반으로 한국어 콘텐츠 여부 판별"""
    if language_code and language_code.startswith("ko"):
        return True
    korean_chars = len(_KOREAN_RE.findall(title + channel))
    total_chars = len(title + channel) or 1
    return (korean_chars / total_chars) > 0.3


class ResearchAgent(BaseAgent):
    """
    외국어(영어 포함) YouTube 동영상을 검색하고 transcript를 저장합니다.

    출력:
      01_research/videos.json
      01_research/transcripts/{video_id}_transcript.txt
      01_research/transcripts/{video_id}_transcript.json
    """

    def run(self) -> dict:
        console.rule("[bold cyan]1단계: 소재 리서치 (Research Agent) — 외국어 영상 전용")

        api_key = os.getenv("YOUTUBE_API_KEY", "")
        if api_key and not api_key.startswith("AIza-your"):
            console.print("  [green]YouTube Data API v3 사용[/green]")
            result = self._search_youtube(api_key)
        else:
            console.print("  [yellow]YOUTUBE_API_KEY 미설정 → LLM 시뮬레이션 모드[/yellow]")
            result = self._simulate_with_llm()

        # transcript 다운로드
        result = self._download_transcripts(result)

        self.write_json("01_research/videos.json", result)
        self._print_ranking_table(result["videos"])
        return result

    # ------------------------------------------------------------------ #
    # YouTube Data API v3 실제 검색                                        #
    # ------------------------------------------------------------------ #
    def _search_youtube(self, api_key: str) -> dict:
        from googleapiclient.discovery import build

        queries = self.settings["research"]["youtube_search_queries"]
        excluded_langs = self.settings["research"].get("excluded_languages", ["ko"])
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
                relevanceLanguage="en",   # 영어 관련성 우선
            ).execute()

            video_ids = [
                item["id"]["videoId"]
                for item in search_resp.get("items", [])
                if item["id"]["videoId"] not in seen_ids
            ]
            seen_ids.update(video_ids)

            if not video_ids:
                continue

            stats_resp = youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=",".join(video_ids),
            ).execute()

            for item in stats_resp.get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                lang = snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage") or ""

                # 한국어 콘텐츠 제외
                if _is_korean_content(snippet.get("title", ""), snippet.get("channelTitle", ""), lang):
                    console.print(f"  [dim]제외 (한국어): {snippet.get('title', '')[:40]}[/dim]")
                    continue

                raw_videos.append({
                    "video_id": item["id"],
                    "title": snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                    "language": lang or "en",
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", "")[:300],
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "search_query": query,
                    "has_transcript": False,
                    "transcript_language": None,
                })

        ranked = self._rank_videos(raw_videos, top_n)
        trend_note = "(YouTube API 검색 결과 기반, transcript 분석 전)"
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "search_queries": queries,
            "excluded_languages": excluded_langs,
            "total_found": len(raw_videos),
            "videos": ranked,
            "trend_analysis": trend_note,
        }

    # ------------------------------------------------------------------ #
    # LLM 시뮬레이션 (API 키 없을 때)                                        #
    # ------------------------------------------------------------------ #
    def _simulate_with_llm(self) -> dict:
        queries = self.settings["research"]["youtube_search_queries"]
        top_n = self.settings["research"]["top_videos_to_rank"]
        system_prompt = self.load_template("research_prompt.txt")
        user_prompt = f"""다음 검색 키워드로 YouTube에서 외국어(영어/기타 외국어) 한국 생활 영상을 검색한다고 가정하고,
한국어가 아닌 외국어로 제작된 인기 영상 {top_n}개 목록을 만들어 주세요.
채널명과 제목이 영어 또는 기타 외국어여야 합니다.

검색 키워드:
{chr(10).join(f'- {q}' for q in queries)}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "searched_at": "ISO8601 날짜",
  "search_queries": ["키워드1"],
  "excluded_languages": ["ko"],
  "total_found": {top_n},
  "videos": [
    {{
      "rank": 1,
      "video_id": "가상ID",
      "title": "영어 제목",
      "channel": "영어 채널명",
      "language": "en",
      "published_at": "YYYY-MM-DDTHH:MM:SSZ",
      "description": "영어 설명 (200자 이내)",
      "url": "https://www.youtube.com/watch?v=가상ID",
      "views": 조회수,
      "likes": 좋아요수,
      "comments": 댓글수,
      "search_query": "해당 검색어",
      "main_topics": ["주제1"],
      "ranking_score": 0.95,
      "ranking_reason": "이유",
      "has_transcript": false,
      "transcript_language": null
    }}
  ],
  "trend_analysis": "트렌드 분석"
}}"""
        raw = self.call_llm(system_prompt, user_prompt)
        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    # Transcript 다운로드                                                   #
    # ------------------------------------------------------------------ #
    def _download_transcripts(self, result: dict) -> dict:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

        preferred_langs = self.settings["research"].get(
            "preferred_transcript_languages", ["en", "en-US", "en-GB"]
        )
        transcript_dir = self.episode_dir / "01_research" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n  [bold]Transcript 다운로드 시작...[/bold]")
        success, skip, fail = 0, 0, 0
        ytt = YouTubeTranscriptApi()

        for video in result.get("videos", []):
            vid_id = video.get("video_id", "")
            if not vid_id or vid_id.startswith("가상") or len(vid_id) < 5:
                video["has_transcript"] = False
                video["transcript_language"] = None
                skip += 1
                continue

            try:
                # 영어 자막 우선, 없으면 자동 생성 자막
                transcript_list = ytt.list(vid_id)

                transcript = None
                used_lang = None
                # 수동 자막 영어 우선
                for lang in preferred_langs:
                    try:
                        transcript = transcript_list.find_transcript([lang])
                        used_lang = lang
                        break
                    except Exception:
                        pass

                # 없으면 자동 생성 자막 (영어)
                if transcript is None:
                    try:
                        transcript = transcript_list.find_generated_transcript(["en"])
                        used_lang = "en-auto"
                    except Exception:
                        pass

                # 없으면 첫 번째 자막 사용 (한국어 제외)
                if transcript is None:
                    for t in transcript_list:
                        if not t.language_code.startswith("ko"):
                            transcript = t
                            used_lang = t.language_code
                            break

                if transcript is None:
                    raise NoTranscriptFound(vid_id, [], {})

                entries = transcript.fetch()
                # 텍스트 파일로 저장
                full_text = " ".join(e.text for e in entries)
                txt_path = transcript_dir / f"{vid_id}_transcript.txt"
                txt_path.write_text(full_text, encoding="utf-8")

                # JSON (타임스탬프 포함)
                json_path = transcript_dir / f"{vid_id}_transcript.json"
                json_path.write_text(
                    json.dumps(
                        {"video_id": vid_id, "language": used_lang,
                         "entries": [{"text": e.text, "start": e.start, "duration": e.duration} for e in entries]},
                        ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

                video["has_transcript"] = True
                video["transcript_language"] = used_lang
                console.print(f"  [green]✓[/green] [{video.get('rank','?')}위] {video.get('title','')[:40]} ({used_lang})")
                success += 1

            except (NoTranscriptFound, TranscriptsDisabled):
                video["has_transcript"] = False
                video["transcript_language"] = None
                console.print(f"  [yellow]—[/yellow] [{video.get('rank','?')}위] {video.get('title','')[:40]} (자막 없음)")
                fail += 1
            except Exception as e:
                video["has_transcript"] = False
                video["transcript_language"] = None
                console.print(f"  [red]✗[/red] [{video.get('rank','?')}위] {video.get('title','')[:40]}: {e}")
                fail += 1

        console.print(f"\n  [bold]Transcript 결과:[/bold] 성공 {success}개 / 자막없음 {fail}개 / 건너뜀 {skip}개")
        result["transcript_stats"] = {"success": success, "no_transcript": fail, "skipped": skip}
        return result

    # ------------------------------------------------------------------ #
    # 랭킹 계산                                                             #
    # ------------------------------------------------------------------ #
    def _rank_videos(self, videos: list[dict], top_n: int) -> list[dict]:
        weights = self.settings["research"]["ranking_weights"]
        now = datetime.now(timezone.utc)
        max_views = max((v["views"] for v in videos), default=1) or 1
        max_likes = max((v["likes"] for v in videos), default=1) or 1

        for v in videos:
            try:
                pub = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00"))
                days_old = (now - pub).days
                recency = max(0.0, 1.0 - days_old / 730)
            except Exception:
                recency = 0.5

            v["ranking_score"] = round(
                weights["views"] * math.log1p(v["views"]) / math.log1p(max_views)
                + weights["likes"] * math.log1p(v["likes"]) / math.log1p(max_likes)
                + weights["recency"] * recency
                + weights["relevance"] * 0.8,
                4
            )

        ranked = sorted(videos, key=lambda x: x["ranking_score"], reverse=True)[:top_n]
        for i, v in enumerate(ranked, 1):
            v["rank"] = i
        return ranked

    # ------------------------------------------------------------------ #
    # 출력 테이블                                                            #
    # ------------------------------------------------------------------ #
    def _print_ranking_table(self, videos: list[dict]) -> None:
        table = Table(title="📊 외국어 한국 생활 인기 동영상 랭킹 (한국어 제외)", show_lines=True)
        table.add_column("순위", style="bold yellow", width=4)
        table.add_column("제목 (외국어)", style="cyan", max_width=38)
        table.add_column("채널", style="green", max_width=18)
        table.add_column("언어", width=6)
        table.add_column("조회수", justify="right", width=9)
        table.add_column("📄", width=3)

        for v in videos[:15]:
            table.add_row(
                str(v.get("rank", "?")),
                v.get("title", "")[:38],
                v.get("channel", "")[:18],
                v.get("language", "")[:6],
                f"{v.get('views', 0):,}",
                "✓" if v.get("has_transcript") else "—",
            )
        console.print(table)

    # ------------------------------------------------------------------ #
    # Mock 응답 (base_agent 위임)                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)



class ResearchAgent(BaseAgent):
    """
    외국인 한국 생활 관련 YouTube 동영상을 검색하고 랭킹을 매깁니다.

    출력:
      01_research/videos.json
      01_research/transcripts/{video_id}_transcript.txt
      01_research/transcripts/{video_id}_transcript.json
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

        # transcript 다운로드
        result = self._download_transcripts(result)

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
                videoDuration="medium",   # 4~20분 영상만 조회
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

            # 통계 + 영상 길이 정보 가져오기
            stats_resp = youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=",".join(video_ids),
            ).execute()

            for item in stats_resp.get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                content = item.get("contentDetails", {})

                duration_iso = content.get("duration", "")
                duration_secs = _parse_duration_seconds(duration_iso)

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
                    "duration_seconds": duration_secs,
                    "duration_iso": duration_iso,
                    "has_transcript": False,
                    "transcript_language": None,
                })

        ranked = self._rank_videos(raw_videos, top_n)
        trend_analysis = self._analyze_trends(ranked)
        return {
            "searched_at": datetime.now(timezone.utc).isoformat(),
            "search_queries": queries,
            "video_duration_filter": "medium",
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
    # Transcript 다운로드                                                   #
    # ------------------------------------------------------------------ #
    def _download_transcripts(self, result: dict) -> dict:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

        preferred_langs = self.settings["research"].get(
            "preferred_transcript_languages", ["en", "en-US", "en-GB"]
        )
        transcript_dir = self.episode_dir / "01_research" / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n  [bold]Transcript 다운로드 시작...[/bold]")
        success, skip, fail = 0, 0, 0
        ytt = YouTubeTranscriptApi()

        for video in result.get("videos", []):
            vid_id = video.get("video_id", "")
            if not vid_id or vid_id.startswith("가상") or len(vid_id) < 5:
                video["has_transcript"] = False
                video["transcript_language"] = None
                skip += 1
                continue

            try:
                transcript_list = ytt.list(vid_id)

                transcript = None
                used_lang = None
                for lang in preferred_langs:
                    try:
                        transcript = transcript_list.find_transcript([lang])
                        used_lang = lang
                        break
                    except Exception:
                        pass

                if transcript is None:
                    try:
                        transcript = transcript_list.find_generated_transcript(["en"])
                        used_lang = "en-auto"
                    except Exception:
                        pass

                if transcript is None:
                    for t in transcript_list:
                        if not t.language_code.startswith("ko"):
                            transcript = t
                            used_lang = t.language_code
                            break

                if transcript is None:
                    raise NoTranscriptFound(vid_id, [], {})

                entries = transcript.fetch()
                full_text = " ".join(e.text for e in entries)

                txt_path = transcript_dir / f"{vid_id}_transcript.txt"
                txt_path.write_text(full_text, encoding="utf-8")

                json_path = transcript_dir / f"{vid_id}_transcript.json"
                json_path.write_text(
                    json.dumps(
                        {"video_id": vid_id, "language": used_lang,
                         "entries": [{"text": e.text, "start": e.start, "duration": e.duration} for e in entries]},
                        ensure_ascii=False, indent=2,
                    ),
                    encoding="utf-8",
                )

                video["has_transcript"] = True
                video["transcript_language"] = used_lang
                console.print(
                    f"  [green]✓[/green] [{video.get('rank', '?')}위] "
                    f"{video.get('title', '')[:40]} ({used_lang})"
                )
                success += 1

            except (NoTranscriptFound, TranscriptsDisabled):
                video["has_transcript"] = False
                video["transcript_language"] = None
                console.print(
                    f"  [yellow]—[/yellow] [{video.get('rank', '?')}위] "
                    f"{video.get('title', '')[:40]} (자막 없음)"
                )
                fail += 1
            except Exception as e:
                video["has_transcript"] = False
                video["transcript_language"] = None
                console.print(
                    f"  [red]✗[/red] [{video.get('rank', '?')}위] "
                    f"{video.get('title', '')[:40]}: {e}"
                )
                fail += 1

        console.print(
            f"\n  [bold]Transcript 결과:[/bold] "
            f"성공 {success}개 / 자막없음 {fail}개 / 건너뜀 {skip}개"
        )
        result["transcript_stats"] = {"success": success, "no_transcript": fail, "skipped": skip}
        return result

    # ------------------------------------------------------------------ #
    # 결과 출력                                                             #
    # ------------------------------------------------------------------ #
    def _print_ranking_table(self, videos: list[dict]) -> None:
        table = Table(title="📊 외국인 한국 생활 인기 동영상 랭킹", show_lines=True)
        table.add_column("순위", style="bold yellow", width=4)
        table.add_column("제목", style="cyan", max_width=38)
        table.add_column("채널", style="green", max_width=18)
        table.add_column("길이", justify="right", width=7)
        table.add_column("조회수", justify="right", width=10)
        table.add_column("📄", width=3)

        for v in videos[:15]:
            secs = v.get("duration_seconds", 0)
            duration_str = f"{secs // 60}:{secs % 60:02d}" if secs else "—"
            table.add_row(
                str(v.get("rank", "?")),
                v.get("title", "")[:38],
                v.get("channel", "")[:18],
                duration_str,
                f"{v.get('views', 0):,}",
                "✓" if v.get("has_transcript") else "—",
            )
        console.print(table)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)


