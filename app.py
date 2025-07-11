from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import norm
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

def black_scholes_and_greeks(S, K, T, r, v):
    """
    Calculates Black-Scholes option prices and their corresponding Greeks.
    This function is vectorized to handle numpy array inputs correctly.

    Args:
        S (float or np.ndarray): Current asset price(s)
        K (float or np.ndarray): Strike price(s)
        T (float): Time to maturity in years
        r (float): Risk-free interest rate (as a decimal)
        v (float or np.ndarray): Volatility of the asset (as a decimal)

    Returns:
        dict: A dictionary containing prices and Greeks.
    """
    # If time is zero, prices are intrinsic values and greeks are mostly zero or undefined.
    if np.isclose(T, 0):
        return {
            'call_price': np.maximum(0, S - K), 'put_price': np.maximum(0, K - S),
            'call_delta': 1.0 if S > K else 0.0, 'put_delta': -1.0 if S < K else 0.0,
            'gamma': 0, 'vega': 0, 'call_theta': 0, 'put_theta': 0
        }

    # --- Standard Black-Scholes Calculation ---
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)

    # --- Prices ---
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # --- Greeks ---
    # Probability density function of the standard normal distribution
    pdf_d1 = norm.pdf(d1)
    
    # Delta
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma
    gamma = pdf_d1 / (S * v * np.sqrt(T))
    
    # Vega (price change per 1% change in volatility)
    vega = S * pdf_d1 * np.sqrt(T) / 100
    
    # Theta (price change per calendar day)
    call_theta = (-(S * pdf_d1 * v) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-(S * pdf_d1 * v) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Handle cases where volatility is zero
    is_v_zero = np.isclose(v, 0)
    if np.any(is_v_zero):
        # For zero vol, price is discounted intrinsic value. Greeks have specific values.
        call_price = np.where(is_v_zero, np.maximum(0, S - K * np.exp(-r * T)), call_price)
        put_price = np.where(is_v_zero, np.maximum(0, K * np.exp(-r * T) - S), put_price)
        call_delta = np.where(is_v_zero, 1.0 if S > K else 0.0, call_delta)
        put_delta = np.where(is_v_zero, -1.0 if S < K else 0.0, put_delta)
        gamma = np.where(is_v_zero, 0, gamma)
        vega = np.where(is_v_zero, 0, vega)
        # Theta at zero vol is simply the interest accruing/decaying on the strike
        call_theta = np.where(is_v_zero, r * K * np.exp(-r * T) / 365, call_theta)
        put_theta = np.where(is_v_zero, r * K * np.exp(-r * T) / 365, put_theta)

    return {
        'call_price': call_price, 'put_price': put_price,
        'call_delta': call_delta, 'put_delta': put_delta,
        'gamma': gamma, 'vega': vega,
        'call_theta': call_theta, 'put_theta': put_theta
    }

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()

    S_base = float(data['assetPrice'])
    K = float(data['strikePrice'])
    T = float(data['timeToMaturity'])
    v_base = float(data['volatility']) / 100
    r = float(data['interestRate']) / 100

    # 1. Calculate single point values including Greeks
    main_values = black_scholes_and_greeks(S_base, K, T, r, v_base)

    # 2. Generate data for the heatmaps (prices only)
    spot_prices = np.linspace(S_base * 0.8, S_base * 1.2, 20)
    volatilities = np.linspace(v_base * 0.5, v_base * 1.5, 20)
    S_grid, v_grid = np.meshgrid(spot_prices, volatilities)
    # Call function for prices only for heatmap
    heatmap_prices = black_scholes_and_greeks(S_grid, K, T, r, v_grid)
    
    call_prices_grid = heatmap_prices['call_price']
    put_prices_grid = heatmap_prices['put_price']

    call_heatmap_data, put_heatmap_data = [], []
    for i in range(len(volatilities)):
        for j in range(len(spot_prices)):
            call_heatmap_data.append({'S': spot_prices[j], 'v': volatilities[i], 'price': call_prices_grid[i, j]})
            put_heatmap_data.append({'S': spot_prices[j], 'v': volatilities[i], 'price': put_prices_grid[i, j]})

    # Prepare the response payload, converting all numpy types to standard Python floats
    response_data = {
        'callValue': float(main_values['call_price']),
        'putValue': float(main_values['put_price']),
        'greeks': {
            'call_delta': float(main_values['call_delta']),
            'put_delta': float(main_values['put_delta']),
            'gamma': float(main_values['gamma']),
            'vega': float(main_values['vega']),
            'call_theta': float(main_values['call_theta']),
            'put_theta': float(main_values['put_theta']),
        },
        'callHeatmap': call_heatmap_data,
        'putHeatmap': put_heatmap_data
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
