import csv
import os
import re
import ssl
from collections import defaultdict
from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
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
DATA_DIR = PROJECT_ROOT / "data" / "C3"
FINANCIALS_PATH = DATA_DIR / "excel" / "Financials.csv"
HIERARCHICAL_DIR = DATA_DIR / "hierarchical"

QUERY = "Mexico 的折扣策略有哪些風險？請根據財務資料與風險備忘錄回答。"
TOP_SOURCE_K = 2
TOP_DOC_K = 3


def clean_header(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def parse_money(value: str) -> float:
    text = value.strip().replace("$", "").replace(",", "").replace(" ", "")
    if text in {"", "-", "-   "}:
        return 0.0
    return float(text)


def load_financial_rows() -> list[dict[str, str]]:
    with FINANCIALS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append({clean_header(key): value.strip() for key, value in row.items()})
    return rows


def build_financial_documents() -> list[Document]:
    rows = load_financial_rows()

    by_country: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_segment: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_country[row["country"]].append(row)
        by_product[row["product"]].append(row)
        by_segment[row["segment"]].append(row)

    documents: list[Document] = []

    def make_summary(group_name: str, group_type: str, group_rows: list[dict[str, str]]) -> Document:
        total_units = sum(parse_money(row["units_sold"]) for row in group_rows)
        total_sales = sum(parse_money(row["sales"]) for row in group_rows)
        total_discounts = sum(parse_money(row["discounts"]) for row in group_rows)
        total_profit = sum(parse_money(row["profit"]) for row in group_rows)
        products = sorted({row["product"] for row in group_rows})
        segments = sorted({row["segment"] for row in group_rows})
        countries = sorted({row["country"] for row in group_rows})
        years = sorted({row["year"] for row in group_rows})

        text = (
            f"{group_type} summary: {group_name}\n"
            f"Rows: {len(group_rows)}\n"
            f"Total units sold: {total_units:,.2f}\n"
            f"Total sales: {total_sales:,.2f}\n"
            f"Total discounts: {total_discounts:,.2f}\n"
            f"Total profit: {total_profit:,.2f}\n"
            f"Products: {', '.join(products)}\n"
            f"Segments: {', '.join(segments)}\n"
            f"Countries: {', '.join(countries)}\n"
            f"Years: {', '.join(years)}\n"
            "Use this summary to answer finance, sales, discount, country, product, and profit questions."
        )

        return Document(
            text=text,
            metadata={
                "source": str(FINANCIALS_PATH.relative_to(PROJECT_ROOT)),
                "source_id": "financials",
                "source_type": "csv",
                "group_type": group_type,
                "group_name": group_name,
            },
        )

    for country, group_rows in by_country.items():
        documents.append(make_summary(country, "country", group_rows))
    for product, group_rows in by_product.items():
        documents.append(make_summary(product, "product", group_rows))
    for segment, group_rows in by_segment.items():
        documents.append(make_summary(segment, "segment", group_rows))

    return documents


def build_markdown_documents(path: Path, source_id: str) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^## ", text)
    documents: list[Document] = []

    for index, section in enumerate(sections):
        content = section.strip()
        if not content:
            continue
        if index == 0:
            section_title = path.stem
            section_text = content
        else:
            lines = content.splitlines()
            section_title = lines[0].strip()
            section_text = "## " + content

        documents.append(
            Document(
                text=section_text,
                metadata={
                    "source": str(path.relative_to(PROJECT_ROOT)),
                    "source_id": source_id,
                    "source_type": "markdown",
                    "section": section_title,
                },
            )
        )

    return documents


def build_source_indexes() -> dict[str, VectorStoreIndex]:
    source_documents = {
        "financials": build_financial_documents(),
        "product_faq": build_markdown_documents(
            HIERARCHICAL_DIR / "product_faq.md",
            "product_faq",
        ),
        "regional_strategy": build_markdown_documents(
            HIERARCHICAL_DIR / "regional_strategy.md",
            "regional_strategy",
        ),
        "risk_memo": build_markdown_documents(
            HIERARCHICAL_DIR / "risk_memo.md",
            "risk_memo",
        ),
    }

    return {
        source_id: VectorStoreIndex.from_documents(documents)
        for source_id, documents in source_documents.items()
    }


def build_router_index() -> VectorStoreIndex:
    router_documents = [
        Document(
            text=(
                "Financials dataset. Use this source for quantitative questions "
                "about sales, units sold, discounts, COGS, profit, country, product, segment, month, and year."
            ),
            metadata={"source_id": "financials", "source_type": "csv"},
        ),
        Document(
            text=(
                "Product FAQ. Use this source for product descriptions and product-level interpretation "
                "for Carretera, Montana, Paseo, VTT, Velo, and Amarilla."
            ),
            metadata={"source_id": "product_faq", "source_type": "markdown"},
        ),
        Document(
            text=(
                "Regional strategy notes. Use this source for country-level strategy, market behavior, "
                "and regional interpretation for Canada, France, Germany, Mexico, and United States."
            ),
            metadata={"source_id": "regional_strategy", "source_type": "markdown"},
        ),
        Document(
            text=(
                "Sales risk memo. Use this source for risk analysis about discounts, segment behavior, "
                "product risk, seasonal effects, and data quality problems."
            ),
            metadata={"source_id": "risk_memo", "source_type": "markdown"},
        ),
    ]
    return VectorStoreIndex.from_documents(router_documents)


def retrieve_from_hierarchical_index(query: str) -> None:
    Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-m3")
    Settings.llm = None

    router_index = build_router_index()
    source_indexes = build_source_indexes()

    router_retriever = router_index.as_retriever(similarity_top_k=TOP_SOURCE_K)
    selected_sources = router_retriever.retrieve(query)

    print("查詢問題:")
    print(query)

    print("\n第一層：選出最相關的資料源")
    for source_rank, source_node in enumerate(selected_sources, start=1):
        source_id = source_node.node.metadata["source_id"]
        print(f"\nSource {source_rank}: {source_id}")
        print(f"score: {source_node.score}")
        print(source_node.node.get_content())

    print("\n第二層：進入被選中的資料源查詢")
    for source_rank, source_node in enumerate(selected_sources, start=1):
        source_id = source_node.node.metadata["source_id"]
        retriever = source_indexes[source_id].as_retriever(similarity_top_k=TOP_DOC_K)
        results = retriever.retrieve(query)

        print(f"\n=== {source_rank}. {source_id} ===")
        for result_rank, result in enumerate(results, start=1):
            node = result.node
            print(f"\nResult {result_rank}")
            print(f"score: {result.score}")
            print("metadata:")
            print(node.metadata)
            print("content:")
            print(node.get_content())


if __name__ == "__main__":
    retrieve_from_hierarchical_index(QUERY)

