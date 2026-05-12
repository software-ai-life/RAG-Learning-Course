import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "data" / "C3" / "images" / "02"


def get_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 GEMINI_API_KEY。請在專案根目錄建立 .env，並加入：\n"
            "GEMINI_API_KEY=your_gemini_api_key_here"
        )

    return api_key


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    raise ValueError(f"不支援的圖片格式：{path.name}")


def build_contents(query: str, image_items: list[dict[str, Path]]) -> list[types.Content]:
    contents = [
        types.Content(
            parts=[
                types.Part.from_text(
                    text=f"task: search result | query: {query}"
                )
            ]
        )
    ]

    for item in image_items:
        image_path = item["path"]
        if not image_path.exists():
            raise FileNotFoundError(f"找不到圖片：{image_path}")

        contents.append(
            types.Content(
                parts=[
                    types.Part.from_bytes(
                        data=image_path.read_bytes(),
                        mime_type=get_mime_type(image_path),
                    )
                ]
            )
        )

    return contents


def main() -> None:
    client = genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            # 教學環境若遇到公司代理或本機自簽憑證，httpx 會拋出
            # CERTIFICATE_VERIFY_FAILED。正式專案建議改成指定可信任 CA。
            httpxClient=httpx.Client(verify=False, follow_redirects=True),
        ),
    )

    image_items = [
        {"title": "RAG 架構圖", "path": IMAGE_DIR / "02_rag_architecture.png"},
        {"title": "AI Agent 圖片", "path": IMAGE_DIR / "02_ai_agent.jpg"},
    ]

    query = "哪張圖片在說明 RAG 的檢索流程？"
    contents = build_contents(query, image_items)

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )

    vectors = [embedding.values for embedding in result.embeddings]
    query_vector = [vectors[0]]
    image_vectors = vectors[1:]

    scores = cosine_similarity(query_vector, image_vectors)[0]

    for item, score in sorted(
        zip(image_items, scores),
        key=lambda item_score: item_score[1],
        reverse=True,
    ):
        print(f"{item['title']}: {score:.4f}")


if __name__ == "__main__":
    main()
