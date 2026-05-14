import os
from glob import glob
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "C3" / "images" / "04_cars"
COLLECTION_NAME = "multimodal_car_demo"
MILVUS_URI = "http://localhost:19530"
EMBEDDING_MODEL = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 768
QUERY_TEXT = "一台白色車"
QUERY_IMAGE_PATH = DATA_DIR / "query.jpg"
RESULT_PATH = DATA_DIR / "search_result.png"


def get_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 GEMINI_API_KEY。請在專案根目錄建立 .env，並加入：\n"
            "GEMINI_API_KEY=your_gemini_api_key_here"
        )

    return api_key


def get_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()

    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    raise ValueError(f"不支援的圖片格式：{path}")


def validate_paths() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "找不到圖片資料夾：\n"
            f"{DATA_DIR}\n\n"
            "請建立 data/C3/images/04_cars/，並放入 query.jpg、car_01.jpg 等圖片。"
        )

    if not QUERY_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 query 圖片：{QUERY_IMAGE_PATH}"
        )


class Encoder:
    def __init__(self) -> None:
        self.client = genai.Client(
            api_key=get_api_key(),
            http_options=types.HttpOptions(
                # 教學環境若遇到公司代理或本機自簽憑證，httpx 會拋出
                # CERTIFICATE_VERIFY_FAILED。正式專案建議改成指定可信任 CA。
                httpxClient=httpx.Client(verify=False, follow_redirects=True),
            ),
        )

    def encode_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_bytes(
                            data=path.read_bytes(),
                            mime_type=get_mime_type(path),
                        )
                    ]
                )
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        return result.embeddings[0].values

    def encode_query(self, image_path: str | Path, text: str) -> list[float]:
        path = Path(image_path)
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text=f"task: search result | query: {text}"
                        ),
                        types.Part.from_bytes(
                            data=path.read_bytes(),
                            mime_type=get_mime_type(path),
                        ),
                    ]
                )
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=OUTPUT_DIMENSIONALITY,
            ),
        )

        return result.embeddings[0].values


def get_image_list() -> list[str]:
    image_patterns = [
        str(DATA_DIR / "*.png"),
        str(DATA_DIR / "*.jpg"),
        str(DATA_DIR / "*.jpeg"),
    ]
    image_list: list[str] = []
    for pattern in image_patterns:
        image_list.extend(glob(pattern))

    query_path = str(QUERY_IMAGE_PATH)
    result_path = str(RESULT_PATH)
    return sorted(
        path
        for path in image_list
        if path not in {query_path, result_path}
    )


def create_collection(client: MilvusClient, dim: int) -> None:
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已刪除既有 Collection：{COLLECTION_NAME}")

    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
    ]
    schema = CollectionSchema(fields, description="車子圖片多模態檢索")
    print("Schema 已定義：", schema)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )
    print(f"已建立 Collection：{COLLECTION_NAME}")
    print(client.describe_collection(collection_name=COLLECTION_NAME))


def insert_images(
    client: MilvusClient,
    encoder: Encoder,
    image_list: list[str],
) -> None:
    data_to_insert = []

    for image_path in tqdm(image_list, desc="產生圖片 embeddings"):
        vector = encoder.encode_image(image_path)
        data_to_insert.append(
            {
                "vector": vector,
                "image_path": image_path,
            }
        )

    if not data_to_insert:
        raise RuntimeError(f"在 {DATA_DIR} 找不到可插入的圖片。")

    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data_to_insert,
    )
    print(f"已插入 {result['insert_count']} 筆圖片資料")


def create_index(client: MilvusClient) -> None:
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 256},
    )

    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )
    client.load_collection(collection_name=COLLECTION_NAME)
    print("已建立 HNSW index，並將 Collection 載入記憶體")


def search_images(client: MilvusClient, encoder: Encoder) -> list[str]:
    query_vector = encoder.encode_query(
        image_path=QUERY_IMAGE_PATH,
        text=QUERY_TEXT,
    )

    search_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        output_fields=["image_path"],
        limit=5,
        search_params={"metric_type": "COSINE", "params": {"ef": 128}},
    )[0]

    retrieved_images = []

    print("\n查詢圖片：")
    print(QUERY_IMAGE_PATH)
    print("查詢文字：")
    print(QUERY_TEXT)
    print("\n檢索結果：")

    for index, hit in enumerate(search_results, start=1):
        image_path = hit["entity"]["image_path"]
        retrieved_images.append(image_path)
        print(
            f"Top {index}: ID={hit['id']}, "
            f"distance={hit['distance']:.4f}, path={image_path}"
        )

    return retrieved_images


def read_and_resize(image_path: str | Path, width: int, height: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_cv = np.array(image)[:, :, ::-1]
    return cv2.resize(image_cv, (width, height))


def visualize_results(
    query_image_path: Path,
    retrieved_images: list[str],
    img_width: int = 260,
    img_height: int = 220,
) -> np.ndarray:
    columns = max(len(retrieved_images), 1)
    canvas_width = img_width * (columns + 1)
    canvas_height = img_height
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    query_image = read_and_resize(query_image_path, img_width, img_height)
    query_image = cv2.copyMakeBorder(
        query_image,
        4,
        4,
        4,
        4,
        cv2.BORDER_CONSTANT,
        value=(255, 0, 0),
    )
    query_image = cv2.resize(query_image, (img_width, img_height))
    canvas[:, 0:img_width] = query_image
    cv2.putText(
        canvas,
        "Query",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    for index, image_path in enumerate(retrieved_images, start=1):
        start_col = index * img_width
        image = read_and_resize(image_path, img_width, img_height)
        image = cv2.copyMakeBorder(
            image,
            2,
            2,
            2,
            2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        image = cv2.resize(image, (img_width, img_height))
        canvas[:, start_col : start_col + img_width] = image
        cv2.putText(
            canvas,
            f"Top {index}",
            (start_col + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return canvas


def main() -> None:
    validate_paths()

    image_list = get_image_list()
    if not image_list:
        raise RuntimeError(
            f"在 {DATA_DIR} 找不到候選圖片。請放入 car_01.jpg、car_02.jpg 等圖片。"
        )

    print("初始化 Gemini embedding encoder 與 Milvus client")
    encoder = Encoder()
    client = MilvusClient(uri=MILVUS_URI)

    dim = len(encoder.encode_image(image_list[0]))
    create_collection(client, dim)
    insert_images(client, encoder, image_list)
    create_index(client)

    retrieved_images = search_images(client, encoder)
    if retrieved_images:
        result_image = visualize_results(QUERY_IMAGE_PATH, retrieved_images)
        cv2.imwrite(str(RESULT_PATH), result_image)
        print(f"\n搜尋結果圖已儲存到：{RESULT_PATH}")

    client.release_collection(collection_name=COLLECTION_NAME)
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"已釋放並刪除 Collection：{COLLECTION_NAME}")


if __name__ == "__main__":
    main()
