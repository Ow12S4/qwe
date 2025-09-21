import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import ta
import requests
from bs4 import BeautifulSoup
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash.dependencies import Output, Input, State
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# 1. Enhanced Settings
# ----------------------------
ticker = 'AAPL'
seq_length = 60  # Increased sequence length for better pattern recognition
future_days = [5, 10, 20]  # Multiple future predictions
simulations = 200  # Increased Monte Carlo simulations
update_interval = 3600*1000  # in milliseconds, 1 hour
model_save_path = 'best_model.h5'
sentiment_analysis_enabled = True
technical_indicators_enabled = True
macro_economic_enabled = True

# ----------------------------
# 2. Enhanced Data Fetching & Feature Engineering
# ----------------------------
def fetch_stock_data(ticker, period='2y', interval='1h'):
    """Fetch stock data with enhanced technical indicators"""
    df = yf.download(ticker, period=period, interval=interval)
    
    if technical_indicators_enabled:
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['Stoch_%K'] = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14).stoch()
        df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Volatility Indicators
        df['BB_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['BB_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume Indicators
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
            df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    
    # Handle missing values more intelligently
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    
    # Create multiple future targets
    for days in future_days:
        df[f'Future_{days}d'] = df['Close'].shift(-days)
        df[f'Movement_{days}d'] = 0
        df.loc[df[f'Future_{days}d'] > df['Close'] * (1 + 0.01 * (days/5)), f'Movement_{days}d'] = 2  # Up
        df.loc[df[f'Future_{days}d'] < df['Close'] * (1 - 0.01 * (days/5)), f'Movement_{days}d'] = 0  # Down
        # Neutral is automatically 1 by default
    
    return df

def fetch_sentiment(ticker):
    """Enhanced sentiment analysis from multiple sources"""
    if not sentiment_analysis_enabled:
        return 0
        
    sentiment_score = 0
    sources = [
        f'https://finance.yahoo.com/quote/{ticker}',
        f'https://www.marketwatch.com/investing/stock/{ticker}',
        f'https://www.bloomberg.com/quote/{ticker}:US'
    ]
    
    positive_keywords = ['profit', 'growth', 'gain', 'beat', 'upgrade', 'buy', 'outperform', 'bullish']
    negative_keywords = ['loss', 'decline', 'drop', 'miss', 'downgrade', 'sell', 'underperform', 'bearish']
    
    for url in sources:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Extract text from relevant elements
            text_content = ' '.join([element.get_text() for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'span'])])
            text_content = text_content.lower()
            
            # Score the sentiment
            for keyword in positive_keywords:
                if keyword in text_content:
                    sentiment_score += 1
                    
            for keyword in negative_keywords:
                if keyword in text_content:
                    sentiment_score -= 1
                    
        except Exception as e:
            print(f"Error fetching sentiment from {url}: {e}")
            continue
    
    # Normalize sentiment score
    return np.tanh(sentiment_score / 10)  # Scale to [-1, 1]

def fetch_macro_economic_indicators():
    """Fetch macroeconomic indicators that might affect stock prices"""
    if not macro_economic_enabled:
        return {}
    
    macro_data = {}
    try:
        # VIX - Market volatility index
        vix = yf.download('^VIX', period='1mo', interval='1d')
        macro_data['vix'] = vix['Close'].iloc[-1] if len(vix) > 0 else 0
        
        # Treasury yields
        treasury_10y = yf.download('^TNX', period='1mo', interval='1d')
        macro_data['treasury_10y'] = treasury_10y['Close'].iloc[-1] if len(treasury_10y) > 0 else 0
        
        # Dollar index
        dollar_index = yf.download('DX-Y.NYB', period='1mo', interval='1d')
        macro_data['dxy'] = dollar_index['Close'].iloc[-1] if len(dollar_index) > 0 else 0
        
    except Exception as e:
        print(f"Error fetching macroeconomic data: {e}")
    
    return macro_data

# ----------------------------
# 3. Enhanced Data Preparation
# ----------------------------
def prepare_data(df, target_days=5):
    """Prepare data for training with enhanced features"""
    # Add sentiment and macroeconomic data
    df['Sentiment'] = fetch_sentiment(ticker)
    macro_data = fetch_macro_economic_indicators()
    
    # Add macroeconomic data as constant columns (will be forward-filled later)
    for key, value in macro_data.items():
        df[key] = value
    
    # Forward fill macroeconomic data
    df.fillna(method='ffill', inplace=True)
    
    # Select features
    base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    technical_features = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'RSI', 
                         'Stoch_%K', 'MACD', 'ADX', 'BB_High', 'BB_Low', 
                         'ATR', 'OBV', 'VWAP'] if technical_indicators_enabled else []
    external_features = ['Sentiment'] + list(macro_data.keys())
    
    features = base_features + technical_features + external_features
    X_data = df[features].values
    
    # Create target based on specified future days
    target_col = f'Movement_{target_days}d'
    y_data = df[target_col].values
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Create sequences with more sophisticated approach
    X, y = [], []
    for i in range(seq_length, len(X_scaled) - max(future_days)):
        X.append(X_scaled[i-seq_length:i])
        y.append(y_data[i])
    
    X, y = np.array(X), np.array(y)
    
    # Convert to categorical (one-hot encoding)
    y_cat = np.zeros((len(y), 3))
    for i, val in enumerate(y):
        y_cat[i, int(val)] = 1
    
    return X, y_cat, scaler, features

# ----------------------------
# 4. Enhanced Model Architecture
# ----------------------------
def build_advanced_model(input_shape):
    """Build a more sophisticated model architecture"""
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.001)), 
                     input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001))),
        Dropout(0.3),
        Bidirectional(GRU(64, kernel_regularizer=l2(0.001))),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam', 
        loss=CategoricalCrossentropy(), 
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# ----------------------------
# 5. Enhanced Training with Validation
# ----------------------------
def train_model(model, X, y, validation_split=0.2):
    """Train model with callbacks and validation"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    ]
    
    history = model.fit(
        X, y, 
        epochs=50, 
        batch_size=32, 
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0
    )
    
    return history, model

# ----------------------------
# 6. Enhanced Monte Carlo Simulation
# ----------------------------
def monte_carlo_prediction(model, X_last, simulations=100):
    """Enhanced Monte Carlo simulation with dropout at inference time"""
    # Enable dropout during inference for uncertainty estimation
    mc_preds = []
    for _ in range(simulations):
        # Use dropout during inference
        pred = model(X_last, training=True).numpy()[0]
        mc_preds.append(pred)
    
    mc_preds = np.array(mc_preds)
    mean_pred = np.mean(mc_preds, axis=0)
    std_pred = np.std(mc_preds, axis=0)
    ci_lower = np.percentile(mc_preds, 5, axis=0)
    ci_upper = np.percentile(mc_preds, 95, axis=0)
    
    return mean_pred, std_pred, ci_lower, ci_upper, mc_preds

# ----------------------------
# 7. Enhanced Dash App
# ----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    html.H1(f"{ticker} Advanced Stock Prediction Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Controls"),
                dbc.CardBody([
                    html.Label("Select Prediction Horizon:"),
                    dcc.Dropdown(
                        id='horizon-selector',
                        options=[{'label': f'{days} days', 'value': days} for days in future_days],
                        value=future_days[0],
                        clearable=False
                    ),
                    html.Br(),
                    dbc.Button("Update Prediction", id="update-button", color="primary", className="mr-1"),
                ])
            ], style={'marginBottom': 20}),
            
            dbc.Card([
                dbc.CardHeader("Current Sentiment & Macro Indicators"),
                dbc.CardBody(id='sentiment-indicators')
            ])
        ], width=3),
        
        dbc.Col([
            dcc.Graph(id='candlestick-graph'),
            dcc.Graph(id='volume-graph'),
            dcc.Graph(id='technical-indicators'),
        ], width=9)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Results"),
                dbc.CardBody(id='movement-output')
            ])
        ])
    ]),
    
    dcc.Interval(id='interval-component', interval=update_interval, n_intervals=0),
    dcc.Store(id='model-store'),
    dcc.Store(id='data-store')
])

@app.callback(
    [Output('data-store', 'data'),
     Output('sentiment-indicators', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('update-button', 'n_clicks')]
)
def update_data(n_intervals, n_clicks):
    """Fetch and prepare data"""
    df = fetch_stock_data(ticker)
    sentiment = fetch_sentiment(ticker)
    macro_data = fetch_macro_economic_indicators()
    
    # Create sentiment indicators display
    sentiment_color = 'success' if sentiment > 0.1 else 'danger' if sentiment < -0.1 else 'warning'
    sentiment_text = html.H4(f"Sentiment: {sentiment:.2f}", 
                            style={'color': 'green' if sentiment > 0 else 'red' if sentiment < 0 else 'orange'})
    
    macro_indicators = [sentiment_text]
    for key, value in macro_data.items():
        macro_indicators.append(html.P(f"{key}: {value:.2f}"))
    
    return df.to_dict('records'), macro_indicators

@app.callback(
    [Output('candlestick-graph', 'figure'),
     Output('volume-graph', 'figure'),
     Output('technical-indicators', 'figure'),
     Output('movement-output', 'children'),
     Output('model-store', 'data')],
    [Input('data-store', 'data'),
     Input('horizon-selector', 'value'),
     Input('update-button', 'n_clicks')],
    [State('model-store', 'data')]
)
def update_dashboard(data_dict, horizon, n_clicks, existing_model):
    """Update all dashboard components"""
    if not data_dict:
        return go.Figure(), go.Figure(), go.Figure(), "Loading data...", None
    
    df = pd.DataFrame.from_dict(data_dict)
    df.index = pd.to_datetime(df.index) if 'Date' not in df.columns else pd.to_datetime(df['Date'])
    
    # Prepare data for the selected horizon
    X, y_cat, scaler, features = prepare_data(df, target_days=horizon)
    if len(X) == 0:
        return go.Figure(), go.Figure(), go.Figure(), "Not enough data to train", None
    
    # Build or load model
    if existing_model:
        try:
            model = load_model(model_save_path)
        except:
            model = build_advanced_model((X.shape[1], X.shape[2]))
    else:
        model = build_advanced_model((X.shape[1], X.shape[2]))
    
    # Train model
    history, model = train_model(model, X, y_cat)
    
    # Predict latest movement
    last_seq = X[-1].reshape(1, seq_length, X.shape[2])
    pred, std, ci_lower, ci_upper, mc_preds = monte_carlo_prediction(model, last_seq, simulations)
    
    # Format prediction results
    movement_map = ['Down', 'Neutral', 'Up']
    movement_text = []
    for i in range(3):
        confidence = std[i]  # Lower std means higher confidence
        text_color = 'red' if i == 0 else 'orange' if i == 1 else 'green'
        movement_text.append(
            html.Li([
                html.Span(f"{movement_map[i]}: ", style={'fontWeight': 'bold'}),
                html.Span(f"{pred[i]*100:.2f}%", style={'color': text_color}),
                html.Span(f" (±{std[i]*100:.2f}%)", style={'fontSize': '0.8em', 'color': 'gray'})
            ])
        )
    
    # Create candlestick chart with predictions
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC'
    )])
    
    # Add technical indicators
    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA_10'], mode='lines', name='SMA(10)'))
    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA(50)'))
    
    # Add future predictions
    last_date = df.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon+1)]
    
    # Create prediction fan chart
    pred_values = np.quantile(mc_preds[:, 2] - mc_preds[:, 0], [0.05, 0.25, 0.5, 0.75, 0.95])
    pred_values = df['Close'].iloc[-1] * (1 + pred_values * 0.01)
    
    fig_candle.add_trace(go.Scatter(
        x=future_dates, 
        y=pred_values,
        fill='tonexty',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name='Prediction Range',
        showlegend=True
    ))
    
    fig_candle.update_layout(
        title=f"{ticker} Price with {horizon}-Day Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=400
    )
    
    # Volume chart
    fig_volume = go.Figure(go.Bar(
        x=df.index, y=df['Volume'],
        marker_color=np.where(df['Close'] > df['Open'], 'green', 'red')
    ))
    fig_volume.update_layout(
        title="Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        height=200
    )
    
    # Technical indicators chart
    fig_tech = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'ADX'),
        vertical_spacing=0.1
    )
    
    # RSI
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=1, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'), row=1, col=2)
    fig_tech.add_hline(y=0, line_dash="dash", line_color="white", row=1, col=2)
    
    # Bollinger Bands
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['BB_High'], mode='lines', name='BB High'), row=2, col=1)
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'), row=2, col=1)
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], mode='lines', name='BB Low'), row=2, col=1)
    
    # ADX
    fig_tech.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX'), row=2, col=2)
    fig_tech.add_hline(y=25, line_dash="dash", line_color="yellow", row=2, col=2)
    
    fig_tech.update_layout(template="plotly_dark", height=400, showlegend=False)
    
    return fig_candle, fig_volume, fig_tech, movement_text, True

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_ui=False)