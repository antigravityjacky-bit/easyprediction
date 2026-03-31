# easyprediction

📂 horseracing_1plus2_core 項目結構說明：
1. horseracing/scraper/ (數據搜刮員 🕷️)
用途：專門用嚟喺馬會 (HKJC) 官網「爬」即時同歷史數據落嚟。
例子：搵今日嘅排位表 (hkjc_race_card.py)、以往嘅賽績同埋步速數據 (hkjc_sectional.py)。
2. horseracing/ml/ (機器學習大腦 🧠)
用途：呢度係模型嘅「諗嘢」邏輯。
例子：將數據砌成 Dataset (dataset.py)，同埋點樣結合 XGB, LGBM 呢啲模型黎做 Ensemble 同權重分配 (ensemble.py)。
3. horseracing/features/ (特徵工程 🛠️)
用途：將原始數據「轉化」做有用嘅指標。
例子：計算馬匹嘅體能爆發力 (physical_performance.py)，或者分析近期嘅賽績趨勢 (engineer.py)。我已經幫你刪走咗 Anti-Trend 嗰段代碼，令佢變返最純淨嘅 1+2 運算。
4. horseracing/prediction/ (預測發布器 📢)
用途：呢度係最終「出結果」嘅窗口。
例子：你主要行嗰個 live_predictor.py 就係喺呢度。佢會讀取所有邏輯，然後印出 Banker 同 Legs 出嚟俾你。
5. horseracing/backtest/ (復盤驗證 🔍)
用途：用嚟測試你套 Logic 喺以前嘅舊賽事入面中唔中。
例子：如果你改咗啲 Formula，你可以用佢黎睇下喺過去幾年會贏定輸。
6. horseracing/constants/ (硬性設定值 ⚙️)
用途：放一啲唔會成日變嘅常數。
例子：例如 ST 代表沙田、HV 代表跑馬地，或者馬場嘅跑道名稱對照表。
7. horseracing/corrections/ (數據清洗員 🧹)
用途：修正一啲馬會數據入面可能出現嘅手誤或者名稱唔統一嘅問題。
8. horseracing/profile/ (馬、騎師檔案室 👥)
用途：建立同儲存馬匹、騎師嘅個人 Profile，方便之後重複讀取。
9. datasets/reference/ (固定基準資料集庫 📚)
用途：放一啲好重要嘅「參考書」。
例子：例如每個路程嘅標準時間 (Par Time)、唔同負磅對馬匹嘅影響、或者唔同配備嘅效果對比。
10. datasets/processed/ (已編譯賽績 💾)
用途：儲存一份總合晒所有歷史賽事嘅大表 (ml_dataset.csv)。
例子：我幫你加咗呢份文件入去，令你新個 Package 唔洗重新「爬」過數據都可以即刻 Predict 到。
11. output/models/ (已訓練模型權重 🚀)
用途：呢度放咗你之前練好嘅 XGB, LGBM 同 LTR 呢幾份「考試成績單」 (.pkl files)。
例子：佢哋就好似係大腦嘅記憶咁，有呢幾份嘢，個程式先可以即刻分辨到邊隻係馬王。
🎁 額外文件：

requirements.txt: 列出晒你需要裝嘅 Python 公具 (Pandas, LightGBM 等等)。
setup.py: 方便將全個 Folder 安裝成一個自定義 Library 嘅設定檔。
