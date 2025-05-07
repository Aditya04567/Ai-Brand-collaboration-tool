import React, { useState } from 'react';

function SearchForm({ setInfluencers, setError, setLoading, disableSubscribers }) {
  const [niche, setNiche] = useState('');
  const [minSubscribers, setMinSubscribers] = useState(1000);
  const [maxSubscribers, setMaxSubscribers] = useState(1000000000);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/influencers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          niche,
          min_subscribers: minSubscribers,
          max_subscribers: maxSubscribers,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setInfluencers(data, null);
      } else {
        throw new Error(data.error || 'Failed to fetch influencers');
      }
    } catch (err) {
      setInfluencers([], err.message);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Niche:</label>
        <input
          type="text"
          value={niche}
          onChange={(e) => setNiche(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter niche or YouTube URL"
        />
      </div>
      <div className="form-group">
        <label>Min Subscribers:</label>
        <input
          type="number"
          value={minSubscribers}
          onChange={(e) => setMinSubscribers(Number(e.target.value))}
          disabled={disableSubscribers}
        />
      </div>
      <div className="form-group">
        <label>Max Subscribers:</label>
        <input
          type="number"
          value={maxSubscribers}
          onChange={(e) => setMaxSubscribers(Number(e.target.value))}
          disabled={disableSubscribers}
        />
      </div>
      <button type="submit" disabled={loading}>
        {loading ? 'Searching...' : 'Search'}
      </button>
    </form>
  );
}

export default SearchForm;