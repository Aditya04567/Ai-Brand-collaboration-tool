import React, { useState, useEffect } from 'react';
import InfluencerCard from './InfluencerCard';

function InfluencerList({ influencers }) {
  alert('InfluencerList component loaded!'); // Debug alert to confirm the component is loaded
  const [sortBy, setSortBy] = useState('performance_score');

  // Log the influencers to debug
  console.log("Influencers received in InfluencerList:", influencers);

  // Preload the first 2 profile pictures for better perceived performance
  useEffect(() => {
    if (influencers && influencers.length > 0) {
      const preloadImages = influencers.slice(0, 2).map((influencer) => {
        if (influencer.profile_picture) {
          const img = new Image();
          img.src = influencer.profile_picture;
          return img;
        }
        return null;
      });
      return () => {
        preloadImages.forEach((img) => {
          if (img) img.src = '';
        });
      };
    }
  }, [influencers]);

  // Handle case where no influencers are found
  if (!influencers || influencers.length === 0) {
    return <p>No influencers found. Try a different niche or adjust the subscriber range.</p>;
  }

  // Sort influencers based on selected criteria
  const sortedInfluencers = [...influencers].sort((a, b) => {
    if (sortBy === 'subscribers') return b.subscribers - a.subscribers;
    if (sortBy === 'engagement_rate') return b.engagement_rate - a.engagement_rate;
    return b.performance_score - a.performance_score;
  });

  return (
    <div>
      <script>alert('List')</script>
      {/* Sorting Controls */}
      <div className="sort-controls" style={{ marginBottom: '20px' }}>
        <label style={{ marginRight: '10px' }}>Sort by: </label>
        <select
          onChange={(e) => setSortBy(e.target.value)}
          value={sortBy}
          style={{ padding: '5px', borderRadius: '4px' }}
        >
          <option value="performance_score">Performance Score</option>
          <option value="subscribers">Subscribers</option>
          <option value="engagement_rate">Engagement Rate</option>
        </select>
      </div>

      {/* Influencer List */}
      <div
        className="influencer-list"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '20px',
        }}
      >
        {sortedInfluencers.map((influencer) => (
          <InfluencerCard key={influencer.channel_id} influencer={influencer} />
        ))}
      </div>
    </div>
  );
}

export default InfluencerList;