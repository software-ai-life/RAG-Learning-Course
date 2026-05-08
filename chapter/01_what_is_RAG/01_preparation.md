# 課前準備

[![English](https://img.shields.io/badge/Language-English-blue)](./01_preparation_en.md)

## 建立虛擬環境

如果還沒有安裝 `uv`，先執行：

```powershell
pip install uv
```

建立並啟動虛擬環境：

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
```

安裝專案需要的套件：

```powershell
uv pip install -r requirements.txt
```

確認 Python 版本與已安裝套件：

```powershell
python --version
uv pip list
```

## AI API Key

這個專案在 `langchain_example.py` 中使用 Gemini 的 OpenAI-compatible endpoint，
所以環境變數名稱必須設定為 `GEMINI_API_KEY`。

### 1. 建立 Gemini API key

1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey)。
2. 使用 Google 帳號登入。
3. 點選 **Create API key**。
4. 選擇既有的 Google Cloud project，或讓 Google AI Studio 幫你建立一個新的 project。
5. 複製產生出來的 API key。

不要把 API key 貼到 Python 檔案、Markdown 檔案、截圖、GitHub 或聊天訊息裡。
請把它當成密碼保管。

### 2. 建立 `.env` 檔案

在專案根目錄建立一個名為 `.env` 的檔案：

```powershell
New-Item -Path .env -ItemType File
```

把你的 Gemini API key 加進 `.env`：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

範例格式如下：

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

上面的 key 只是 placeholder，請替換成你在 Google AI Studio 取得的真實 key。

### 3. 確認 `.env` 不會被提交

確認 `.env` 有被加入 `.gitignore`。如果專案還沒有 `.gitignore`，先建立它：

```powershell
if (-not (Test-Path .gitignore)) {
    New-Item -Path .gitignore -ItemType File
}
```

### 4. 在 Python 中讀取 API key

`requirements.txt` 已經包含 `python-dotenv`。在 Python 裡，先載入 `.env`，
再讀取 `GEMINI_API_KEY`：

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
```

目前的 LangChain 範例已經使用這個 key：

```python
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
```
