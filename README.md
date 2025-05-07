# Influencer Analyzer



## Overview

The Influencer Analyzer is an open-source web application designed to streamline the identification and evaluation of YouTube influencers for marketing campaigns. By integrating the YouTube Data API for real-time data retrieval and the Gemini API for AI-driven insights, it enables users to search influencers by niche and subscriber count, view detailed analytics, and generate comprehensive reports assessing campaign suitability. Built with a React frontend and Flask backend, the application offers a responsive, user-friendly interface and robust functionality for marketers, small business owners, and influencer agencies.

## Features

- **Niche-Based Search:** Filter influencers by niche (e.g., sustainable fashion) and subscriber range.
- **Detailed Analytics:** View influencer profiles with metrics like subscribers, views, engagement rate, and a custom performance score.
- **AI-Generated Reports:** Leverage Gemini API to analyze influencer performance and brand alignment.
- **Interactive Visualizations:** Display engagement trends and keyword frequency using Chart.js.
- **Responsive Design:** Accessible across desktops, tablets, and mobiles with Tailwind CSS.
- **Robust Backend:** Flask-powered API with caching (LRU cache) and error handling.
- **Open-Source:** Fully customizable with community contributions encouraged.

## Tech Stack

### Frontend

- React: Component-based UI with hooks for state management.
- Tailwind CSS: Utility-first styling for responsive design.
- Chart.js: Interactive charts for analytics visualization.
- Axios: Asynchronous API requests.

### Backend

- Flask: Lightweight Python framework for RESTful API.
- Python: Core logic with DistilBERT for sentiment analysis.
- YouTube Data API: Real-time influencer and video data.
- Gemini API: AI-driven report generation.

## Screenshots
![Screenshot 2025-05-06 153600](https://github.com/user-attachments/assets/cc85f914-a0d7-408a-a812-ae5c9c98f36c)
![Screenshot 2025-05-06 153727](https://github.com/user-attachments/assets/60f3529b-77dd-4f36-9281-96e730064972)
![screencapture-127-0-0-1-6969-2025-05-06-15_38_15](https://github.com/user-attachments/assets/16abc21e-bc3a-49d1-8c36-aebdf8f47141)

## Installation

### Prerequisites

- Node.js (>=16.x)
- Python (>=3.8)
- Git
- YouTube Data API key
- Gemini API key

### Steps

#### Clone the Repository:

```bash
git clone https://github.com/Aditya04567/Ai-Brand-collaboration-tool
cd influencer-analyzer
```

#### Backend Setup:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the backend directory:

```
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
FLASK_ENV=development
```

Run the Flask server:

```bash
flask run
```

#### Frontend Setup:

```bash
cd ../frontend
npm install
```

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:5000
```

Start the React development server:

```bash
npm start
```

### Access the Application:

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
influencer-analyzer/
├── backend/
│   ├── app.py              # Flask application entry point
│   ├── influencer_analyzer.py  # Core logic for API integrations
│   ├── requirements.txt    # Python dependencies
│   └── tests/              # Backend unit tests
├── frontend/
│   ├── src/
│   │   ├── components/     # React components (SearchForm, InfluencerList, etc.)
│   │   ├── App.js          # Main React app
│   │   └── index.js        # Entry point
│   ├── package.json        # Node dependencies
│   └── tests/              # Frontend unit tests
├── docs/                   # Documentation and diagrams
└── README.md               # Project documentation
```

## Usage

### Search Influencers:

- Navigate to the homepage.
- Enter a niche (e.g., "sustainable fashion") and subscriber range.
- Click "Search" to view a grid of matching influencers.

### View Profiles:

- Click an influencer card to access detailed analytics, including engagement trends and keyword frequency charts.

### Generate Reports:

- On the profile page, request an AI-generated report to assess campaign suitability.

## Current Status

- Backend: Fully operational with 95% test coverage, caching, and error handling.
- Frontend: Functional components implemented but facing a rendering issue (white page, likely due to React Router/Webpack misconfiguration).
- Testing: Backend tests complete; frontend tests pending resolution of rendering issue.
- Accessibility: 90% WCAG 2.1 compliant, with minor ARIA label issues.

## Challenges

- Frontend Rendering: White page issue requires debugging of React Router or build configuration.
- API Rate Limits: YouTube API quotas necessitate advanced caching strategies.
- Scalability: Backend query optimization needed for large datasets.

## Future Plans

- Resolve frontend rendering issue.
- Integrate Instagram API for multi-platform support.
- Conduct user testing with marketing professionals.


## License

This project is licensed under the MIT License. See the LICENSE file for details.






