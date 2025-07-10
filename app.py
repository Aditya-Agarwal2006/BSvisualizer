from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import norm
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow the frontend to communicate with this server
CORS(app)

def black_scholes(S, K, T, r, v):
    """
    Calculates the Black-Scholes option prices for a call and a put.
    This function is vectorized to handle numpy array inputs correctly.

    Args:
        S (float or np.ndarray): Current asset price(s)
        K (float or np.ndarray): Strike price(s)
        T (float): Time to maturity in years
        r (float): Risk-free interest rate (as a decimal)
        v (float or np.ndarray): Volatility of the asset (as a decimal)

    Returns:
        dict: A dictionary containing the call and put prices.
    """
    # If time to maturity is zero, the option price is its intrinsic value.
    if np.isclose(T, 0):
        call_price = np.maximum(0, S - K)
        put_price = np.maximum(0, K - S)
        return {'call': call_price, 'put': put_price}

    # --- Standard Black-Scholes Calculation (for v > 0) ---
    with np.errstate(divide='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
        d2 = d1 - v * np.sqrt(T)
    
    call_standard = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_standard = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # --- Zero Volatility Case (for v = 0) ---
    call_zero_vol = np.maximum(0, S - K * np.exp(-r * T))
    put_zero_vol = np.maximum(0, K * np.exp(-r * T) - S)

    # --- Combine using np.where ---
    is_v_zero = np.isclose(v, 0)
    call_price = np.where(is_v_zero, call_zero_vol, call_standard)
    put_price = np.where(is_v_zero, put_zero_vol, put_standard)

    return {'call': call_price, 'put': put_price}

@app.route('/calculate', methods=['POST'])
def calculate():
    """
    API endpoint to perform Black-Scholes calculations.
    Receives parameters as JSON and returns results as JSON.
    """
    data = request.get_json()

    S_base = float(data['assetPrice'])
    K = float(data['strikePrice'])
    T = float(data['timeToMaturity'])
    v_base = float(data['volatility']) / 100
    r = float(data['interestRate']) / 100

    # 1. Calculate single call/put values
    main_prices = black_scholes(S_base, K, T, r, v_base)

    # 2. Generate data for the heatmaps
    spot_prices = np.linspace(S_base * 0.8, S_base * 1.2, 20)
    volatilities = np.linspace(v_base * 0.5, v_base * 1.5, 20)
    
    S_grid, v_grid = np.meshgrid(spot_prices, volatilities)
    heatmap_prices = black_scholes(S_grid, K, T, r, v_grid)
    
    call_prices_grid = heatmap_prices['call']
    put_prices_grid = heatmap_prices['put']

    call_heatmap_data = []
    put_heatmap_data = []
    for i in range(len(volatilities)):
        for j in range(len(spot_prices)):
            call_heatmap_data.append({
                'S': spot_prices[j],
                'v': volatilities[i],
                'price': call_prices_grid[i, j]
            })
            put_heatmap_data.append({
                'S': spot_prices[j],
                'v': volatilities[i],
                'price': put_prices_grid[i, j]
            })

    # Prepare the response payload
    # **FIX:** Convert NumPy types to standard Python floats before sending as JSON.
    response_data = {
        'callValue': float(main_prices['call']),
        'putValue': float(main_prices['put']),
        'callHeatmap': call_heatmap_data,
        'putHeatmap': put_heatmap_data
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
