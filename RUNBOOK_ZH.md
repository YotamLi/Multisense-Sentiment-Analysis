# MultiSense V2 最终运行手册

这是最终 V2 单版本项目，不包含旧五分类模型兼容路线。

## 新电脑首次安装

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

复制 `.env.example` 为 `.env`，并填写：

```env
GOOGLE_API_KEY=你的Gemini_API_Key
```

最终权重放在：

```text
checkpoints/best_model.pt
```

不要复制旧电脑的 `.venv`；虚拟环境必须在新电脑重新创建。

## 运行最终网页

```powershell
python app\demo.py --config config\config.yaml --checkpoint checkpoints\best_model.pt
```

## 重新构建 RAG

```powershell
python scripts\build_rag.py --config config\config.yaml
```

## 重新生成和验证数据

```powershell
python scripts\prepare_multisource_data.py --config config\config.yaml
python scripts\audit_dataset.py --data-dir data\processed_v2 --output reports\dataset_audit_v2.json
python scripts\validate_dataset.py --data-dir data\processed_v2
```

## Smoke Training

```powershell
python scripts\create_smoke_dataset.py
python scripts\train.py --config config\config_smoke.yaml --output-dir checkpoints\smoke
```

## 正式训练和评估

```powershell
python scripts\train.py --config config\config.yaml --output-dir checkpoints
python scripts\evaluate.py --config config\config.yaml --checkpoint checkpoints\best_model.pt
```

## GitHub 不上传

- `.env`
- `.venv/`
- `checkpoints/best_model.pt`
- `data/processed_v2/`
- `data/chroma_db/`

模型权重建议上传到 GitHub Release 或 Hugging Face Model Repository。
