import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os

engine = create_async_engine(os.environ["DATABASE_URL"])

async def load_data() -> pd.DataFrame:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT child_id, start, \"end\" FROM sleep_events"))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df

def to_seconds(dt_series):
    return dt_series.dt.hour * 3600 + dt_series.dt.minute * 60 + dt_series.dt.second

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'start_seconds_total': to_seconds(df['start']),
        'start_hour':          df['start'].dt.hour,
        'start_minute':        df['start'].dt.minute,
        'day_of_week':         df['start'].dt.dayofweek,
    })

def seconds_to_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
# Простая сеть: 4 признака → 64 → 32 → 1
class EndTimeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


async def predict_end(child_id: int, start_str: str) -> str:
    df = await load_data()

    X_df = extract_features(df)
    y_np = to_seconds(df['end']).values.astype(np.float32)
    weights_np = df['child_id'].apply(
        lambda uid: 5.0 if uid == child_id else 1.0
    ).values.astype(np.float32)

    # Нормализация — для нейросети обязательна, для sklearn нет
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_np = scaler_X.fit_transform(X_df).astype(np.float32)
    y_scaled = scaler_y.fit_transform(y_np.reshape(-1, 1)).flatten().astype(np.float32)

    X_tensor = torch.tensor(X_np)
    y_tensor = torch.tensor(y_scaled)
    w_tensor = torch.tensor(weights_np)

    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EndTimeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction='none')  # reduction='none' чтобы применить веса вручную

    model.train()
    for epoch in range(100):
        for X_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = (loss_fn(preds, y_batch) * w_batch).mean()
            loss.backward()
            optimizer.step()

    # Предсказание
    dt = pd.to_datetime(start_str)
    X_pred_df = pd.DataFrame([{
        'start_seconds_total': dt.hour * 3600 + dt.minute * 60 + dt.second,
        'start_hour':          dt.hour,
        'start_minute':        dt.minute,
        'day_of_week':         dt.dayofweek,
    }])

    X_pred_np = scaler_X.transform(X_pred_df).astype(np.float32)
    X_pred_tensor = torch.tensor(X_pred_np)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_pred_tensor).item()

    pred_seconds = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return seconds_to_time(pred_seconds)
