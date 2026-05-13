# ReviewPulse CN Pro Demo

这是一个面向门店运营者的评论风险识别 Demo，支持：

- 固定 baseline 风险结构分析
- 第 n 期 vs baseline 对比
- 第 n 期 vs 第 n-1 期对比
- 新问题识别
- 风险优先级判断
- 中文整改建议

## 部署到 Streamlit Community Cloud

1. 新建一个 GitHub 仓库。
2. 上传以下文件到仓库根目录：
   - `app.py`（将 `app_deploy_ready.py` 改名为 `app.py`）
   - `baseline.xlsx`
   - `requirements.txt`
   - `README.md`
   - `sample_data/2026_period_1_demo.xlsx`
   - `sample_data/2026_period_2_demo.xlsx`（可选）
3. 登录 Streamlit Community Cloud。
4. 选择你的 GitHub 仓库，并指定入口文件为 `app.py`。
5. 点击 Deploy，生成公开访问链接。

## 建议的仓库结构

```text
reviewpulse-demo/
├─ app.py
├─ baseline.xlsx
├─ requirements.txt
├─ README.md
└─ sample_data/
   ├─ 2026_period_1_demo.xlsx
   └─ 2026_period_2_demo.xlsx
```

## Demo 使用方式

打开链接后，评审可以：

- 直接点击“**一键载入第1期样例演示**”快速查看效果；
- 或先下载样例数据，再体验手动上传流程；
- 也可以上传自己的评论数据集进行分析。

## 数据说明

- `baseline.xlsx`：固定基线，不会在运行中自动更新。
- `sample_data/*.xlsx`：用于演示上传和对比分析流程。
