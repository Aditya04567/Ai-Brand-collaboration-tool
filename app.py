from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from influencer_analyzer import lmao, find_influencers_in_niche, analyze_influencer, predict_future_performance, extract_niche_keywords, generate_ai_channel_report
import os
import logging
from flask import Flask, request, jsonify
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import logging
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/')
def index():
    """Serve the main page with React app."""
    lmao()
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return jsonify({'error': 'Failed to load the application'}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    try:
        response = send_from_directory('static', path)
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
    except Exception as e:
        logger.error(f"Error serving static file {path}: {e}")
        return jsonify({'error': 'Static file not found'}), 404

@app.route('/api/influencers', methods=['POST'])
def get_influencers():
    """API endpoint to fetch influencers based on user input."""
    try:
        data = request.get_json()
        niche = data.get('niche', '')
        min_subs = int(data.get('min_subscribers', 0))
        max_subs = int(data.get('max_subscribers', float('inf'))) if data.get('max_subscribers') else float('inf')

        if not niche:
            logger.warning("Niche parameter missing in request")
            return jsonify({'error': 'Niche is required'}), 400

        logger.info(f"Fetching influencers for niche: {niche}, min_subs: {min_subs}, max_subs: {max_subs}")
        influencers = find_influencers_in_niche(niche, min_subs, max_subs)
        return jsonify(influencers)
    except Exception as e:
        logger.error(f"Error fetching influencers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/influencer/<channel_id>', methods=['GET'])
def get_influencer_details(channel_id):
    """API endpoint to fetch detailed analytics for a specific channel."""
    try:
        min_subs = 0  # Since the channel is already selected, no need to filter
        max_subs = float('inf')
        niche = request.args.get('niche', '')
        niche_keywords = extract_niche_keywords([])  # Pass relevant videos if available
        logger.info(f"Fetching details for channel_id: {channel_id}")
        influencer = analyze_influencer(channel_id, min_subs, max_subs, niche_keywords)
        if not influencer:
            logger.warning(f"Channel not found or does not meet criteria: {channel_id}")
            return jsonify({'error': 'Channel not found or does not meet criteria'}), 404
        return jsonify(influencer)
    except Exception as e:
        logger.error(f"Error fetching influencer details for {channel_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/influencer/<channel_id>/report', methods=['GET'])
def generate_influencer_report(channel_id):
    """API endpoint to generate an AI-driven report for a specific channel using Gemini API."""
    try:
        min_subs = 0
        max_subs = float('inf')
        niche = request.args.get('niche', '')
        niche_keywords = extract_niche_keywords([])  # Pass relevant videos if available
        logger.info(f"Generating AI report for channel_id: {channel_id}")
        influencer = analyze_influencer(channel_id, min_subs, max_subs, niche_keywords)
        
        if not influencer:
            logger.warning(f"Channel not found or does not meet criteria: {channel_id}")
            return jsonify({'error': 'Channel not found or does not meet criteria'}), 404

        # Generate AI-driven report using Gemini API
        report = generate_ai_channel_report(influencer)
        return jsonify({'report': report})
    except Exception as e:
        logger.error(f"Error generating report for {channel_id}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6969))
    app.run(debug=False, host='0.0.0.0', port=port)