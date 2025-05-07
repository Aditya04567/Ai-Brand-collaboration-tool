import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SearchForm from './components/SearchForm';
import InfluencerList from './components/InfluencerList';
import DetailedAnalysis from './components/DetailedAnalysis';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [influencers, setInfluencers] = useState([]);
  const [filteredInfluencers, setFilteredInfluencers] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sortOption, setSortOption] = useState('performance_score');
  const [viralKeywordFilter, setViralKeywordFilter] = useState('');

  // Callback to handle search results
  const handleSearch = (newInfluencers, newError) => {
    setLoading(false);
    if (newError) {
      setError(newError);
      setInfluencers([]);
      setFilteredInfluencers([]);
      setViralKeywordFilter('');
    } else {
      setError(null);
      setInfluencers(newInfluencers);
      setFilteredInfluencers(newInfluencers);
      setViralKeywordFilter('');
    }
  };

  // Reset the search results
  const handleReset = () => {
    setInfluencers([]);
    setFilteredInfluencers([]);
    setError(null);
    setLoading(false);
    setSortOption('performance_score');
    setViralKeywordFilter('');
  };

  // Handle sorting of influencers
  const handleSortChange = (e) => {
    const option = e.target.value;
    setSortOption(option);

    const sortedInfluencers = [...filteredInfluencers].sort((a, b) => {
      if (option === 'subscribers') {
        return b.subscribers - a.subscribers;
      } else if (option === 'engagement_rate') {
        return b.engagement_rate - a.engagement_rate;
      } else if (option === 'keyword_score') {
        return b.keyword_score - a.keyword_score;
      } else {
        return b.performance_score - a.performance_score;
      }
    });

    setFilteredInfluencers(sortedInfluencers);
  };

  // Handle filtering by viral keywords
  const handleViralKeywordFilter = (e) => {
    const keyword = e.target.value;
    setViralKeywordFilter(keyword);

    if (keyword === '') {
      setFilteredInfluencers(influencers);
    } else {
      const filtered = influencers.filter((influencer) =>
        influencer.videos.some((video) =>
          video.text_content.toLowerCase().includes(keyword.toLowerCase())
        )
      );
      setFilteredInfluencers(filtered);
    }
  };

  // Prepare data for viral keyword distribution chart
  const viralKeywordData = influencers.length > 0 &&
    influencers[0].top_viral_keywords &&
    influencers[0].top_viral_keywords.length > 0 ? {
    labels: influencers[0].top_viral_keywords,
    datasets: [
      {
        label: 'Number of Viral Videos',
        data: influencers[0].top_viral_keywords.map(
          (keyword) => influencers[0].keyword_counts[keyword] || 0
        ),
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        borderColor: 'rgba(255, 159, 64, 1)',
        borderWidth: 1,
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Number of Viral Videos',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Viral Keywords',
        },
      },
    },
  };

  return (
    <Router>
      <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{ textAlign: 'center', marginBottom: '20px' }}>Influencer Analytics Dashboard</h1>

        <Routes>
          <Route
            path="/"
            element={
              <>
                {/* Search Form and Reset Button */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '20px',
                  }}
                >
                  <SearchForm setInfluencers={handleSearch} setError={setError} setLoading={setLoading} />
                  {influencers.length > 0 && (
                    <button
                      onClick={handleReset}
                      style={{
                        padding: '8px 16px',
                        backgroundColor: '#dc3545',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                      }}
                    >
                      Reset Search
                    </button>
                  )}
                </div>

                {/* Error Message */}
                {error && (
                  <p style={{ color: 'red', textAlign: 'center', marginBottom: '20px' }}>{error}</p>
                )}

                {/* Main Content */}
                {loading ? (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <p>Loading influencers...</p>
                  </div>
                ) : (
                  <>
                    {influencers.length > 0 && (
                      <>
                        {/* Sort Controls */}
                        <div style={{ marginBottom: '20px' }}>
                          <label style={{ marginRight: '10px' }}>Sort by:</label>
                          <select
                            value={sortOption}
                            onChange={handleSortChange}
                            style={{ padding: '5px', borderRadius: '4px' }}
                          >
                            <option value="performance_score">Performance Score</option>
                            <option value="subscribers">Subscribers</option>
                            <option value="engagement_rate">Engagement Rate</option>
                            <option value="keyword_score">Keyword Score</option>
                          </select>
                        </div>

                        {/* Filter by Viral Keywords */}
                        <div style={{ marginBottom: '20px' }}>
                          <label style={{ marginRight: '10px' }}>Filter by Viral Keyword:</label>
                          <select
                            value={viralKeywordFilter}
                            onChange={handleViralKeywordFilter}
                            style={{ padding: '5px', borderRadius: '4px' }}
                          >
                            <option value="">All Keywords</option>
                            {influencers[0].top_viral_keywords && influencers[0].top_viral_keywords.length > 0 ? (
                              influencers[0].top_viral_keywords.map((keyword) => (
                                <option key={keyword} value={keyword}>
                                  {keyword}
                                </option>
                              ))
                            ) : (
                              <option value="" disabled>No viral keywords available</option>
                            )}
                          </select>
                        </div>

                        {/* Display Top Viral Keywords Across All Channels */}
                        {influencers[0].top_viral_keywords && influencers[0].top_viral_keywords.length > 0 && (
                          <div style={{ marginBottom: '20px' }}>
                            <h3 style={{ fontSize: '1.2rem' }}>Top Viral Keywords Across Channels</h3>
                            <p>
                              <strong>Keywords:</strong> {influencers[0].top_viral_keywords.join(', ')}
                            </p>
                            <p>{influencers[0].viral_keyword_analysis}</p>
                          </div>
                        )}

                        {/* Viral Keyword Distribution Chart */}
                        {viralKeywordData && (
                          <div style={{ marginBottom: '20px', height: '300px' }}>
                            <h3 style={{ fontSize: '1.2rem' }}>Viral Keyword Distribution</h3>
                            <Bar data={viralKeywordData} options={chartOptions} />
                          </div>
                        )}

                        {/* Influencer List */}
                        <InfluencerList influencers={filteredInfluencers} />
                      </>
                    )}
                  </>
                )}
              </>
            }
          />
          <Route path="/influencer/:channel_id" element={<DetailedAnalysis />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;