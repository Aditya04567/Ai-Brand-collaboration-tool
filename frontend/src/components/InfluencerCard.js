import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const InfluencerCard = ({ influencer }) => {
  alert('InfluencerCard component loaded!'); // Debug alert to confirm the component is loaded
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Debug log to confirm the component is rendering
  console.log('Rendering InfluencerCard for:', influencer?.name, 'Channel ID:', influencer?.channel_id);

  const handleGenerateReport = async () => {
    console.log('Generate Report button clicked for:', influencer?.channel_id);
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`/api/influencer/${influencer.channel_id}/report`);
      if (!response.data.report) {
        throw new Error('No report data received from the server');
      }
      setReport(response.data.report);
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message || 'Failed to generate report. Please try again.';
      setError(errorMessage);
      console.error('Error generating report:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = () => {
    console.log('View Details button clicked for:', influencer?.channel_id);
    navigate(`/api/influencer/${influencer.channel_id}`);
  };

  return (
    <div
      className="influencer-card"
      style={{
        border: '1px solid #ddd',
        padding: '16px',
        margin: '10px',
        borderRadius: '8px',
        backgroundColor: '#fff',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      }}
    >
      <img
        src={influencer.profile_picture}
        alt={influencer.name}
        style={{ width: '100px', height: '100px', borderRadius: '50%', objectFit: 'cover' }}
      />
      <h3 style={{ fontSize: '1.2rem', margin: '10px 0' }}>{influencer.name}</h3>
      <p>Subscribers: {influencer.subscribers.toLocaleString()}</p>
      <p>Engagement Rate: {influencer.engagement_rate.toFixed(2)}%</p>
      <p>Performance Score: {influencer.performance_score.toFixed(2)}</p>
        <button
          onClick={handleViewDetails}
          style={{
            padding: '8px 16px',
            backgroundColor: '#007bff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background-color 0.2s',
          }}
          onMouseEnter={(e) => (e.target.style.backgroundColor = '#0056b3')}
          onMouseLeave={(e) => (e.target.style.backgroundColor = '#007bff')}
          aria-label={`View details for ${influencer.name}`}
        >
          View Details
        </button>
        <button
          onClick={handleGenerateReport}
          disabled={loading}
          style={{
            padding: '8px 16px',
            backgroundColor: loading ? '#ccc' : '#28a745',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            transition: 'background-color 0.2s',
            display: 'inline-block', // Ensure button is not hidden
            visibility: 'visible', // Ensure button is visible
          }}
          onMouseEnter={(e) => !loading && (e.target.style.backgroundColor = '#218838')}
          onMouseLeave={(e) => !loading && (e.target.style.backgroundColor = '#28a745')}
          aria-label={`Generate report for ${influencer.name}`}
        >
          {loading ? 'Generating...' : 'Generate Report'}
        </button>
      {error && <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>}
      {report && (
        <div style={{ marginTop: '20px', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px' }}>
          <h4 style={{ fontSize: '1.1rem', marginBottom: '10px' }}>Channel Report</h4>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>{report}</pre>
        </div>
      )}
    </div>
  );
};

export default InfluencerCard;
