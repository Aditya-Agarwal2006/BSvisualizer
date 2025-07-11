import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly
import plotly.graph_objects as go

st.set_page_config(
    page_title="Black-Scholes Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Mathematical Functions ---
def black_scholes_and_greeks(S, K, T, r, v):
    
    if np.isclose(T, 0):
        call_price = np.maximum(0, S - K)
        put_price = np.maximum(0, K - S)
        call_delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        put_delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        gamma = 0
        vega = 0
        call_theta = -r * K * np.exp(-r * T) if S > K else 0
        put_theta = -r * K * np.exp(-r * T) if S < K else 0
        return {
            'call_price': call_price, 'put_price': put_price,
            'call_delta': call_delta, 'put_delta': put_delta,
            'gamma': gamma, 'vega': vega,
            'call_theta': call_theta, 'put_theta': put_theta
        }

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    gamma = pdf_d1 / (S * v * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    call_theta = (-(S * pdf_d1 * v) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-(S * pdf_d1 * v) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


    is_v_zero = np.isclose(v, 0)
    if np.any(is_v_zero):
        call_price = np.where(is_v_zero, np.maximum(0, S - K * np.exp(-r * T)), call_price)
        put_price = np.where(is_v_zero, np.maximum(0, K * np.exp(-r * T) - S), put_price)
        
        call_delta = np.where(is_v_zero, 1.0 if S > K else 0.0, call_delta)
        put_delta = np.where(is_v_zero, -1.0 if S < K else 0.0, put_delta)
        gamma = np.where(is_v_zero, 0, gamma)
        vega = np.where(is_v_zero, 0, vega)

    return {
        'call_price': call_price, 'put_price': put_price,
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'vega': vega,
        'call_theta': call_theta, 'put_theta': put_theta
    }


st.sidebar.header("Input Parameters")

st.sidebar.text_input("Stock Ticker (for reference)", "AAPL")

S = st.sidebar.number_input("Current Asset Price ($)", min_value=0.1, value=150.0, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", min_value=0.1, value=155.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, value=0.25, step=0.01)
v = st.sidebar.slider("Volatility (%)", min_value=1, max_value=100, value=25, step=1)
r = st.sidebar.slider("Risk-Free Interest Rate (%)", min_value=0, max_value=20, value=2, step=1)


v_dec = v / 100.0
r_dec = r / 100.0

# --- Main Panel ---
st.title("📈 Black-Scholes Option Pricing Visualizer")
st.markdown("An interactive tool to understand option pricing, risk (the Greeks), and profit/loss scenarios.")

values = black_scholes_and_greeks(S, K, T, r_dec, v_dec)

st.header("Calculated Values")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Call Option Price", value=f"${values['call_price']:.2f}")
with col2:
    st.metric(label="Put Option Price", value=f"${values['put_price']:.2f}")

st.subheader("Option Greeks (Risk Sensitivities)")
gcol1, gcol2 = st.columns(2)
with gcol1:
    st.markdown("#### Call Greeks")
    st.text(f"Delta (Δ):   {values['call_delta']:.4f}")
    st.text(f"Gamma (Γ):   {values['gamma']:.4f}")
    st.text(f"Vega (ν):    {values['vega']:.4f}")
    st.text(f"Theta (Θ):   {values['call_theta']:.4f}")
with gcol2:
    st.markdown("#### Put Greeks")
    st.text(f"Delta (Δ):   {values['put_delta']:.4f}")
    st.text(f"Gamma (Γ):   {values['gamma']:.4f}")
    st.text(f"Vega (ν):    {values['vega']:.4f}")
    st.text(f"Theta (Θ):   {values['put_theta']:.4f}")


st.header("Profit & Loss Visualizer")
pnl_col1, pnl_col2 = st.columns([1, 2])
with pnl_col1:
    trade_type = st.radio("Trade Type", ["Call", "Put"])
    trade_price = st.number_input("Your Trade Price ($)", min_value=0.0, value=round(values['call_price'] if trade_type == "Call" else values['put_price'], 2), step=0.01)

spot_range = np.linspace(S * 0.7, S * 1.3, 100)
if trade_type == "Call":
    intrinsic_value = np.maximum(0, spot_range - K)
else: # Put
    intrinsic_value = np.maximum(0, K - spot_range)
pnl = intrinsic_value - trade_price

with pnl_col2:
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=spot_range, y=pnl, mode='lines', name='P&L', line=dict(color='cyan', width=3)))
    fig_pnl.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig_pnl.update_layout(
        title=f"{trade_type} Option P&L at Expiration",
        xaxis_title="Spot Price at Expiry ($)",
        yaxis_title="Profit / Loss ($)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_pnl, use_container_width=True)


st.header("Price Sensitivity Heatmaps")
h_col1, h_col2 = st.columns(2)

spot_prices = np.linspace(S * 0.8, S * 1.2, 20)
volatilities = np.linspace(v_dec * 0.5, v_dec * 1.5, 20)
S_grid, v_grid = np.meshgrid(spot_prices, volatilities)

heatmap_values = black_scholes_and_greeks(S_grid, K, T, r_dec, v_grid)
call_prices_grid = heatmap_values['call_price']
put_prices_grid = heatmap_values['put_price']


with h_col1:
    fig_call_hm = go.Figure(data=go.Heatmap(
        z=call_prices_grid,
        x=spot_prices,
        y=volatilities * 100,
        colorscale='Greens',
        hovertemplate='Spot: %{x:.2f}<br>Volatility: %{y:.2f}%<br>Price: %{z:.2f}<extra></extra>'
    ))
    fig_call_hm.update_layout(
        title="Call Price vs. Spot & Volatility",
        xaxis_title="Spot Price ($)",
        yaxis_title="Volatility (%)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_call_hm, use_container_width=True)


with h_col2:
    fig_put_hm = go.Figure(data=go.Heatmap(
        z=put_prices_grid,
        x=spot_prices,
        y=volatilities * 100,
        colorscale='Reds',
        hovertemplate='Spot: %{x:.2f}<br>Volatility: %{y:.2f}%<br>Price: %{z:.2f}<extra></extra>'
    ))
    fig_put_hm.update_layout(
        title="Put Price vs. Spot & Volatility",
        xaxis_title="Spot Price ($)",
        yaxis_title="Volatility (%)",
        template="plotly_dark"
    )
    st.plotly_chart(fig_put_hm, use_container_width=True)
