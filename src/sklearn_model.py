import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
import os

engine = create_async_engine(os.environ["DATABASE_URL"])

async def load_data() -> pd.DataFrame:
    async with engine.connect() as conn:
        result = await conn.execute(text('SELECT child_id, "start", "end" FROM sleep_events WHERE "start" >= (NOW() - INTERVAL \'1 month\')::text'))
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


async def predict_end(child_id: int, start_str: str) -> str:
    df = await load_data()

    X_np = extract_features(df).values
    y_np = to_seconds(df['end']).values.astype(np.float64)
    weights = df['child_id'].apply(
        lambda uid: 5.0 if uid == child_id else 1.0
    ).values

    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    model.fit(X_np, y_np, sample_weight=weights)

    dt = pd.to_datetime(start_str)
    X_pred = np.array([[
        dt.hour * 3600 + dt.minute * 60 + dt.second,
        dt.hour,
        dt.minute,
        dt.dayofweek,
    ]])

    pred_seconds = model.predict(X_pred)[0]
    return seconds_to_time(pred_seconds)
