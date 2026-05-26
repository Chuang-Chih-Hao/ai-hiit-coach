# Render 部署設定

## Build Command
```bash
pip install -r requirements.txt
```

## Start Command
```bash
gunicorn app:app
```

## Environment Variables
- `GROQ_API_KEY`: 你的 Groq API Key

## 注意
- Render 雲端沒有伺服器端實體攝像頭，所以本版本使用瀏覽器 `getUserMedia()` 取得使用者本機攝像頭。
- 前端會定時把 canvas JPEG frame POST 到 `/analyze_frame`。
- 後端只做 MediaPipe 姿態分析並回傳 JSON。
