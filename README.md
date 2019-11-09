# Deep-Time-Series-Prediction
SOTA DeepLearning Models for Time Series Prediction and implemented by Keras and Pytorch.

## Models

- [x] Seq2Seq
- [x] Wavenet
- [ ] GANs

## Examples

### Arima Curve Prediction

Example code in [1_Use_SimpleSeq2Seq_SimpleWaveNet_for_arima_curve_prediction](/notebooks/1_Use_SimpleSeq2Seq_SimpleWaveNet_for_arima_curve_prediction.ipynb)

![](/assets/1_arima_curve.png)

- seq2seq

![seq2seq 1](/assets/1_seq2seq_pred_0.png)

![seq2seq 2](/assets/1_seq2seq_pred_39.png)

- wavenet

![wavenet 1](/assets/1_wavenet_pred_0.png)

![wavenet 2](/assets/1_wavenet_pred_39.png)

## Tricks

- Walk Forward Split
- Windows Data Augments
- [SMAC3](https://automl.github.io/SMAC3/stable/) Hyperparameter Optimizer
- Hard encode the long lagged data
- [COCOB optimizer](https://arxiv.org/abs/1705.07795)
- Checkpoint & Seed Ensemble

## Refs

- [WaveNet Keras Toturial: TimeSeries_Seq2Seq](https://github.com/JEddy92/TimeSeries_Seq2Seq)
- [WaveNet Kaggle Web Traffic Forcasting Competition RANK 6](https://github.com/sjvasquez/web-traffic-forecasting)
- [Seq2Seq Kaggle Web Traffic Forcasting Competition RANK 1](https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795#latest-631996)
- [Kaggle: Corporación Favorita Grocery Sales Forecasting Top1 LSTM/LGBM](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47582)
- [Kaggle: Corporación Favorita Grocery Sales Forecasting Top5 LGBM/CNN/Seq2Seq](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47556)
- [Temporal Pattern Attention for Multivariate Time Series Forecasting, 2018](https://arxiv.org/abs/1809.04206)


