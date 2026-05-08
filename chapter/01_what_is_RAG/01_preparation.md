# 課前準備 / Preparation

> 中文版為主要內容。若需要英文版，請展開下方的 **English Version**。

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

接著檢查 `.gitignore` 裡是否已經有 `.env`：

```powershell
Select-String -Path .gitignore -Pattern "^\.env$"
```

如果沒有任何結果，就把 `.env` 加進去：

```powershell
Add-Content -Path .gitignore -Value ".env"
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

### 5. 選用：只在目前 PowerShell session 設定 key

如果不想建立 `.env` 檔案，也可以在目前的 PowerShell session 暫時設定：

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

這個設定只會在目前的終端機視窗有效。關掉終端機後，需要重新設定。

### 6. 確認 API key 是否成功載入

在專案根目錄執行：

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GEMINI_API_KEY loaded:', bool(os.getenv('GEMINI_API_KEY')))"
```

如果輸出是 `GEMINI_API_KEY loaded: True`，代表 API key 已經成功載入。

<details>
<summary>English Version</summary>

## Create Virtual Environment

Install `uv` if needed:

```powershell
pip install uv
```

Create and activate the virtual environment:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

Check the environment:

```powershell
python --version
uv pip list
```

## AI API Key

This project uses Gemini through the OpenAI-compatible endpoint in
`langchain_example.py`, so the environment variable name must be
`GEMINI_API_KEY`.

### 1. Create a Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Choose an existing Google Cloud project, or let Google AI Studio create one
   for you.
5. Copy the generated API key.

Do not paste the API key into Python files, Markdown files, screenshots, GitHub,
or chat messages. Treat it like a password.

### 2. Create a `.env` file

Create a file named `.env` in the project root:

```powershell
New-Item -Path .env -ItemType File
```

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

For example, the file should look like this:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The example above is only a placeholder. Replace it with your own key from
Google AI Studio.

### 3. Make sure `.env` is not committed

Make sure `.env` is listed in `.gitignore`. If `.gitignore` does not exist yet,
this command will create it:

```powershell
if (-not (Test-Path .gitignore)) {
    New-Item -Path .gitignore -ItemType File
}
```

Then check whether `.env` is already listed:

```powershell
Select-String -Path .gitignore -Pattern "^\.env$"
```

If there is no result, add it:

```powershell
Add-Content -Path .gitignore -Value ".env"
```

### 4. Load the API key in Python

The dependency `python-dotenv` is already included in `requirements.txt`. In
Python, load the `.env` file before reading the environment variable:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
```

The current LangChain example already uses this key:

```python
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
```

### 5. Optional: set the key only for the current PowerShell session

If you do not want to create a `.env` file, you can set the key temporarily in
PowerShell:

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```

This only works for the current terminal session. After closing the terminal,
you need to set it again.

### 6. Verify the key is available

Run this command from the project root:

```powershell
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GEMINI_API_KEY loaded:', bool(os.getenv('GEMINI_API_KEY')))"
```

If the output is `GEMINI_API_KEY loaded: True`, the key has been loaded
successfully.

</details>
