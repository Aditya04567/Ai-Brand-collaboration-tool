# Influencer Analyzer

## Project Description
Influencer Analyzer is a web application that helps users discover and analyze influencers within specific niches. The backend is built with Flask and leverages AI and machine learning models to analyze influencer performance and generate AI-driven reports. The frontend is a React application that provides an interactive user interface for searching influencers, viewing detailed analytics, and accessing AI-generated reports.

## Features
- Search influencers by niche and subscriber count range
- View detailed analytics for individual influencer channels
- Generate AI-driven performance reports for influencers using advanced models
- Responsive React frontend with charts and interactive components
- Integration with Google APIs for enhanced data access

## Backend Setup

### Prerequisites
- Python 3.8 or higher
- Virtual environment tool (venv or virtualenv)

### Installation
1. Clone the repository and navigate to the backend directory:
   ```bash
   cd venv/influencer_analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   # On Windows
   .\env\Scripts\activate
   # On macOS/Linux
   source env/bin/activate
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Backend
Start the Flask application:
```bash
python app.py
```
The backend server will run on `http://0.0.0.0:6969` by default.

## Frontend Setup

### Prerequisites
- Node.js (version 16 or higher recommended)
- npm (comes with Node.js)

### Installation
1. Navigate to the frontend directory:
   ```bash
   cd venv/influencer_analyzer/frontend
   ```

2. Install the dependencies:
   ```bash
   npm install
   ```

### Running the Frontend
Start the development server:
```bash
npm start
```
The frontend will be available at `http://localhost:8080` (default webpack-dev-server port).

### Building for Production
To create a production build:
```bash
npm run build
```

## Usage

- Access the frontend via the development server or production build.
- Use the search form to find influencers by niche and subscriber count.
- Click on an influencer to view detailed analytics and AI-generated reports.
- The backend API endpoints include:
  - `POST /api/influencers` - Fetch influencers by niche and subscriber range.
  - `GET /api/influencer/<channel_id>` - Get detailed analytics for a specific influencer.
  - `GET /api/influencer/<channel_id>/report` - Generate AI-driven report for an influencer.

## Environment Variables and Ports
- Backend runs on port `6969` by default. You can change this by setting the `PORT` environment variable.
- Frontend runs on port `8080` by default via webpack-dev-server.

## License
This project is licensed under the MIT License. (Update as appropriate)

## Author
(Your Name or Organization)
