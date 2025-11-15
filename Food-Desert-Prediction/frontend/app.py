"""
Flask Frontend Application for Food Desert Prediction

Temporary frontend for validation and tweaking predictions.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# API endpoint (default to localhost)
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', api_url=API_BASE_URL)

@app.route('/tract/<tract_id>')
def tract_detail(tract_id):
    """Tract detail page."""
    return render_template('tract_detail.html', tract_id=tract_id, api_url=API_BASE_URL)

@app.route('/state/<state>')
def state_view(state):
    """State view page."""
    return render_template('state_view.html', state=state, api_url=API_BASE_URL)

@app.route('/api/proxy/<path:endpoint>')
def api_proxy(endpoint):
    """Proxy API requests to avoid CORS issues."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        params = request.args.to_dict()
        
        response = requests.get(url, params=params, timeout=10)
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

