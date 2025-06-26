import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# 🔧 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 종목별 경로
file_paths = {
    "AAPL": "C:/Users/blue0/Downloads/AAPL.csv",
    "MSFT": "C:/Users/blue0/Downloads/MSFT.csv",
    "NVDA": "C:/Users/blue0/Downloads/NVDA.csv"
}

# 시계열 데이터 구성 함수
def create_dataset(data, look_back=30):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back, :])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

# ESN 모델 정의
class ESN(nn.Module):
    def __init__(self, input_size, reservoir_size=100, spectral_radius=0.9):
        super(ESN, self).__init__()
        self.Win = torch.empty(reservoir_size, input_size).uniform_(-0.5, 0.5)
        W = torch.empty(reservoir_size, reservoir_size).uniform_(-0.5, 0.5)
        eigvals = torch.linalg.eigvals(W).abs()
        W *= spectral_radius / eigvals.max()
        self.W = W
        self.Wout = nn.Linear(reservoir_size, 1)

    def forward(self, X):
        batch_size, seq_len, _ = X.size()
        h = torch.zeros(batch_size, self.W.shape[0])
        for t in range(seq_len):
            u = X[:, t, :]
            h = torch.tanh(torch.matmul(u, self.Win.T) + torch.matmul(h, self.W.T))
        return self.Wout(h)

# 종목 반복
for stock, path in file_paths.items():
    print(f"\n===== {stock} 예측 시작 =====")

    df = pd.read_csv(path).dropna()
    df.set_index('Date', inplace=True)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, Y = create_dataset(scaled, look_back=30)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

    split = int(len(X_tensor) * 0.8)
    X_train, X_test = X_tensor[:split], X_tensor[split:]
    Y_train, Y_test = Y_tensor[:split], Y_tensor[split:]

    model = ESN(input_size=X_train.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.Wout.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        output = model(X_train)
        loss = criterion(output, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"{stock} 최종 학습 손실: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()
        ground_truth = Y_test.numpy()

    pred_prices = scaler.inverse_transform(
        np.concatenate([predictions, X_test[:, -1, 1:].numpy()], axis=1)
    )[:, 0]
    true_prices = scaler.inverse_transform(
        np.concatenate([ground_truth, X_test[:, -1, 1:].numpy()], axis=1)
    )[:, 0]

    # smoothing
    smooth_pred = pd.Series(pred_prices).rolling(window=5).mean()

    # MAE, RMSE 계산
    mae = mean_absolute_error(true_prices, pred_prices)
    rmse = np.sqrt(mean_squared_error(true_prices, pred_prices))

    print(f"[{stock}] MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, label='실제 종가')
    plt.plot(smooth_pred, label='예측 종가 (이동평균)', color='orange')
    plt.title(f"{stock} 종가 예측 (ESN)")
    plt.xlabel("시간")
    plt.ylabel("가격")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
