import argparse
import os
import ssl
from collections import Counter
from pathlib import Path

import pandas as pd
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# 教學環境若遇到本機憑證或公司代理問題，可先關閉 SSL 驗證。
# 正式專案建議改成設定可信任 CA。
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import requests

    requests.packages.urllib3.disable_warnings()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MUSIC_PATH = PROJECT_ROOT / "data" / "C3" / "excel" / "music.xlsx"

QUERY = "Find 2010 pop songs related to violence and summarize what the lyrics are about."
TOP_YEAR_K = 1
TOP_SONG_K = 5
DEFAULT_ROW_LIMIT = 300


def read_music_sheets(path: Path) -> dict[str, list[dict[str, str]]]:
    excel_file = pd.ExcelFile(path)
    sheets: dict[str, list[dict[str, str]]] = {}

    for sheet_name in excel_file.sheet_names:
        dataframe = pd.read_excel(excel_file, sheet_name=sheet_name)
        dataframe.columns = [
            str(column).strip().lower().replace(" ", "_")
            for column in dataframe.columns
        ]
        dataframe = dataframe.fillna("").astype(str)
        sheets[sheet_name] = dataframe.to_dict("records")

    return sheets


def build_year_router_index(sheets: dict[str, list[dict[str, str]]]) -> VectorStoreIndex:
    year_nodes: list[IndexNode] = []

    for year, rows in sheets.items():
        topics = Counter(row.get("topic", "") for row in rows if row.get("topic"))
        genres = Counter(row.get("genre", "") for row in rows if row.get("genre"))
        top_topics = ", ".join(topic for topic, _ in topics.most_common(6))
        top_genres = ", ".join(genre for genre, _ in genres.most_common(4))

        year_nodes.append(
            IndexNode(
                text=(
                    f"Music dataset for year {year}. "
                    f"This sheet contains {len(rows)} songs. "
                    f"Top topics include {top_topics}. "
                    f"Genres include {top_genres}. "
                    "Use this source for year-specific music, artist, track, genre, topic, and lyrics questions."
                ),
                index_id=f"music_{year}",
                metadata={
                    "source_id": f"music_{year}",
                    "year": year,
                    "source": str(MUSIC_PATH.relative_to(PROJECT_ROOT)),
                    "source_type": "xlsx_sheet",
                },
            )
        )

    return VectorStoreIndex(nodes=year_nodes)


def build_song_retrievers(
    sheets: dict[str, list[dict[str, str]]],
    row_limit: int,
) -> dict[str, object]:
    song_retrievers: dict[str, object] = {}

    for year, rows in sheets.items():
        selected_rows = rows[:row_limit] if row_limit > 0 else rows
        song_documents: list[Document] = []

        for row_number, row in enumerate(selected_rows, start=1):
            text = (
                f"Year: {year}\n"
                f"Artist: {row.get('artist_name', '')}\n"
                f"Track: {row.get('track_name', '')}\n"
                f"Release date: {row.get('release_date', '')}\n"
                f"Genre: {row.get('genre', '')}\n"
                f"Topic: {row.get('topic', '')}\n"
                f"Lyrics keywords: {row.get('lyrics', '')}"
            )

            song_documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_id": f"music_{year}",
                        "source": str(MUSIC_PATH.relative_to(PROJECT_ROOT)),
                        "source_type": "xlsx_song_row",
                        "year": year,
                        "row_number": row_number,
                        "artist_name": row.get("artist_name", ""),
                        "track_name": row.get("track_name", ""),
                        "genre": row.get("genre", ""),
                        "topic": row.get("topic", ""),
                    },
                )
            )

        song_index = VectorStoreIndex.from_documents(song_documents)
        song_retrievers[f"music_{year}"] = song_index.as_retriever(
            similarity_top_k=TOP_SONG_K
        )

    return song_retrievers


def retrieve_from_recursive_index(query: str, row_limit: int) -> None:
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
    Settings.llm = None

    sheets = read_music_sheets(MUSIC_PATH)
    year_router_index = build_year_router_index(sheets)
    song_retrievers = build_song_retrievers(sheets, row_limit=row_limit)

    root_retriever = year_router_index.as_retriever(similarity_top_k=TOP_YEAR_K)
    recursive_retriever = RecursiveRetriever(
        "root",
        retriever_dict={
            "root": root_retriever,
            **song_retrievers,
        },
        verbose=True,
    )

    selected_years = root_retriever.retrieve(query)
    recursive_results = recursive_retriever.retrieve(query)

    print("查詢問題:")
    print(query)
    print(f"\n資料來源: {MUSIC_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "每個年份 sheet 建立索引的列數: "
        f"{row_limit}（row_limit <= 0 代表使用全部資料）"
    )

    print("\n第一層 root retriever 選到的年份 sheet")
    for year_rank, year_node in enumerate(selected_years, start=1):
        source_id = year_node.node.metadata["source_id"]
        print(f"\nYear Source {year_rank}: {source_id}")
        print(f"score: {year_node.score}")
        print(year_node.node.get_content())

    print("\nRecursiveRetriever 最終回傳的歌曲結果")
    for result_rank, result in enumerate(recursive_results, start=1):
        node = result.node
        print(f"\nSong Result {result_rank}")
        print(f"score: {result.score}")
        print("metadata:")
        print(node.metadata)
        print("content:")
        print(node.get_content())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive retrieval example with music.xlsx.")
    parser.add_argument(
        "--row-limit",
        type=int,
        default=DEFAULT_ROW_LIMIT,
        help="Rows per year sheet to index. Use 0 or a negative number for all rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve_from_recursive_index(QUERY, row_limit=args.row_limit)
