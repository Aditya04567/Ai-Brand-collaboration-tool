import React from 'react';
import { Bar } from 'react-chartjs-2';

function DetailedAnalysis({ influencer, onClose }) {
  // Default values to prevent undefined errors
  const {
    name = 'Unknown Influencer',
    channel_url = '#',
    subscribers = 0,
    total_views = 0,
    video_count = 0,
    engagement_rate = 0,
    performance_score = 0,
    keyword_score = 0,
    videos = [],
    top_viral_keywords = [],
    keyword_counts = {},
    prediction = { predicted_engagement_rate: 0, recommendation: 'No prediction available.' },
    brand_collaboration_report = 'No collaboration report available.',
    top_hashtags = [],
    report = 'No report available.',
  } = influencer || {};

  // Prepare data for the engagement chart
  const engagementData = {
    labels: videos.length > 0 ? videos.map((video) => video.title.slice(0, 20) + '...') : ['No Videos'],
    datasets: [
      {
        label: 'Engagement Rate (%)',
        data: videos.length > 0 ? videos.map((video) => video.engagement_rate || 0) : [0],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Engagement Rate (%)',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Recent Videos',
        },
      },
    },
  };

  // Check if there's meaningful data to display
  const hasData = videos.length > 0 || top_viral_keywords.length > 0;

  return (
    <div className="detailed-analysis">
      <h2>{name}</h2>
      <button className="close-button" onClick={onClose}>Close</button>

      {hasData ? (
        <>
          {/* Basic Info */}
          <div className="influencer-info">
            <p><strong>Channel:</strong> <a href={channel_url} target="_blank" rel="noopener noreferrer">{name}</a></p>
            <p><strong>Subscribers:</strong> {subscribers.toLocaleString()}</p>
            <p><strong>Total Views:</strong> {total_views.toLocaleString()}</p>
            <p><strong>Video Count:</strong> {video_count}</p>
            <p><strong>Engagement Rate:</strong> {(engagement_rate).toFixed(2)}%</p>
            <p><strong>Performance Score:</strong> {performance_score.toFixed(2)}</p>
            <p><strong>Keyword Score:</strong> {keyword_score.toFixed(2)}</p>
          </div>

          {/* Engagement Chart */}
          {videos.length > 0 ? (
            <div className="chart-container">
              <h3>Engagement per Video</h3>
              <Bar data={engagementData} options={chartOptions} />
            </div>
          ) : (
            <p>No video engagement data available.</p>
          )}

          {/* Chart Analysis Section */}
          <div className="chart-analysis">
            <h3>Chart Analysis</h3>
            {videos.length > 0 ? (
              <>
                <p>This chart displays the engagement rate for {name}'s recent videos. The engagement rate is calculated as (likes + comments) / views * 100.</p>
                <p><strong>Popular Hashtags:</strong> {top_hashtags.length > 0 ? top_hashtags.map(h => `${h.hashtag} (${h.count})`).join(', ') : 'None'}</p>
              </>
            ) : (
              <p>No engagement data available to analyze.</p>
            )}
          </div>

          {/* Detailed Channel Report */}
          <div className="channel-report">
            <h3>Detailed Channel Report</h3>
            <pre>{report}</pre>
          </div>

          {/* Viral Keywords */}
          {top_viral_keywords.length > 0 ? (
            <div className="viral-keywords">
              <h3>Top Viral Keywords</h3>
              <p>{top_viral_keywords.join(', ')}</p>
            </div>
          ) : (
            <p>No viral keywords available.</p>
          )}

          {/* Brand Collaboration Report */}
          <div className="collaboration-report">
            <h3>Brand Collaboration Report</h3>
            <p>{brand_collaboration_report}</p>
          </div>

          {/* Future Performance Prediction */}
          <div className="prediction">
            <h3>Future Performance Prediction</h3>
            <p><strong>Predicted Engagement Rate:</strong> {(prediction.predicted_engagement_rate).toFixed(2)}%</p>
            <p><strong>Recommendation:</strong> {prediction.recommendation}</p>
          </div>
        </>
      ) : (
        <div className="no-data">
          <p>No Data Available</p>
          <button className="close-button" onClick={onClose}>Close</button>
        </div>
      )}
    </div>
  );
}

export default DetailedAnalysis;