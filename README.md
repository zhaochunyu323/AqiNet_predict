# AqiNet_predict
Predicting AQI of air quality monitoring in Beijing using TF. I divided the map of Beijing into a 10*10 grid map. Then fill the station into the appropriate grid, which leverages the spatial information. In additon, time series contains the temporal information.

The dataset used is from Microsoft Research.

[Yu Zheng, Xiuwen Yi, Ming Li, Ruiyuan Li, Zhangqing Shan, Eric Chang, Tianrui Li. Forecasting Fine-Grained Air Quality Based on Big Data. In the Proceeding of the 21th SIGKDD conference on Knowledge Discovery and Data Mining (KDD 2015)](https://scholar.google.com/scholar_url?url=https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Forecasting20air20qualtiy-kdd2015-camera-ready.pdf&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=9739619300106008042&ei=eezkW6XWFMr7yATeoYjYDQ&scisig=AAGBfm3-XsiI5AuutcAPNolUp5FJalfWNA)

How to train?<br>
---
Using train_merge_tfrecord.py<br>

How to test?<br>
---
Using latest_12.py<br>

weight.py is used when you want to add Adaboost into your train.
