# Preparation

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
