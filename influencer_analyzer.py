import googleapiclient.discovery
import googleapiclient.errors
import re
import statistics
import json
import string
import logging
from collections import Counter
import torch
import numpy as np
from transformers import pipeline, DistilBertTokenizer, DistilBertModel

# New imports for Gemini API
import google.generativeai as genai
import os

# Configure logging (already in your code)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directly set your API key here
GEMINI_API_KEY = "your api key"  # Replace with your actual API key

# Configure Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    gemini_model = None

# Existing code (omitted for brevity - append the new function below your existing code)

def generate_ai_channel_report(influencer_data):
    """
    Generate an in-depth AI-driven report for the channel using Gemini API.
    """
    if not gemini_model:
        logger.error("Gemini API not initialized")
        return "Error: Gemini API not initialized. Please check your API key and configuration."

    try:
        # Prepare channel data as a prompt
        channel_info = (
            f"Channel Name: {influencer_data['name']}\n"
            f"Subscribers: {influencer_data['subscribers']:,}\n"
            f"Total Views: {influencer_data['total_views']:,}\n"
            f"Video Count: {influencer_data['video_count']}\n"
            f"Engagement Rate: {influencer_data['engagement_rate']:.2f}%\n"
            f"Performance 2 Score: {influencer_data['performance_score']:.2f}\n"
            f"Growth Ratio: {influencer_data['growth_ratio']:.2f}%\n"
            f"Average Comments per Video: {influencer_data['avg_comments']:.2f}\n"
            f"Popular Keywords: {', '.join(influencer_data['popular_keywords'])}\n"
            f"Top Viral Keywords: {', '.join(influencer_data['top_viral_keywords'])}\n"
            f"Top Hashtags: {', '.join([h['hashtag'] for h in influencer_data['top_hashtags']])}\n"
            f"Brand Alignment Score: {influencer_data['brand_alignment']['brand_alignment_score']}%\n"
            f"Average Sentiment: {influencer_data['brand_alignment']['average_sentiment']:.2f}\n"
            f"Keyword Match Percentage: {influencer_data['brand_alignment']['keyword_match_percentage']:.2f}%\n"
            f"Prediction: Predicted Engagement Rate: {influencer_data['prediction']['predicted_engagement_rate']}%, "
            f"Recommendation: {influencer_data['prediction']['recommendation']}\n"
        )

        # Prepare video data summary
        video_summary = "Recent Video Insights:\n"
        for video in influencer_data['videos'][:5]:  # Limit to top 5 videos for brevity
            video_summary += (
                f"- Title: {video['title']}\n"
                f"  Views: {video['views']:,}\n"
                f"  Likes: {video['likes']:,}\n"
                f"  Comments: {video['comments']:,}\n"
                f"  Engagement Rate: {video['engagement_rate']:.2f}%\n"
                f"  Sentiment: {video['sentiment_label']} (Score: {video['sentiment_score']:.2f})\n"
            )

        # Combine into a prompt
        prompt = (
            "You are an expert in YouTube channel analytics. Based on the following channel statistics and video insights, "
            "generate a detailed report analyzing the channel's performance, audience engagement, content strategy, and growth potential. "
            "Provide actionable recommendations for improving the channel's performance and alignment with brand values (sustainable fashion and ethical living). "
            "The report should be structured with sections for Overview, Audience Analysis, Content Performance, Growth Opportunities, and Recommendations.\n\n"
            f"{channel_info}\n{video_summary}"
        )

        # Call Gemini API to generate the report
        response = gemini_model.generate_content(prompt)
        report = response.text

        logger.info(f"Generated AI report for channel {influencer_data['name']}")
        return report
    except Exception as e:
        logger.error(f"Error generating AI report: {e}")
        return f"Error generating report: {str(e)}"


# Replace with your YouTube Data API key
YOU_API_KEY = "your api key"


# Initialize YouTube API client
try:
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOU_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize YouTube API client: {e}")
    youtube = None

# Initialize DistilBERT for sentiment analysis
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    logger.error(f"Failed to initialize sentiment analyzer: {e}")
    sentiment_analyzer = None

# Initialize DistilBERT tokenizer and model for keyword extraction
try:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
except Exception as e:
    logger.error(f"Failed to initialize DistilBERT tokenizer/model: {e}")
    tokenizer = None
    distilbert_model = None

# Brand values
BRAND_VALUES = "Promoting sustainable fashion and ethical living"

# Expanded stopwords
STOPWORDS = {
    "the", "is", "in", "and", "to", "of", "a", "for", "with", "on", "at", "by", "from", "up", "about", "into", "over", "after",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "be", "have", "has", "had", "do", "does", "did", "but", "or", "as", "if", "then"
}

# --- Helper Functions ---
def extract_hashtags(text):
    """Extract hashtags from text."""
    try:
        return [tag.lower() for tag in re.findall(r'#\w+', text, re.IGNORECASE)]
    except Exception as e:
        logger.error(f"Error extracting hashtags: {e}")
        return []

# --- Data Fetching Functions ---
def search_channels_by_niche(niche, max_results=3):
    if not youtube:
        logger.error("YouTube API client not initialized")
        return []
    try:
        enhanced_query = f"{niche} influencer OR vlogger OR creator OR channel -inurl:(signup OR login)"
        channels = []
        next_page_token = None
        while len(channels) < max_results:
            request = youtube.search().list(
                part="snippet",
                q=enhanced_query,
                type="channel",
                maxResults=min(50, max_results - len(channels)),
                pageToken=next_page_token,
                relevanceLanguage="en"
            )
            response = request.execute()
            for item in response.get("items", []):
                channels.append({
                    "channel_id": item["snippet"]["channelId"],
                    "name": item["snippet"]["title"],
                    "description": item["snippet"]["description"]
                })
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        logger.info(f"Found {len(channels)} channels for niche: {niche}")
        return channels[:max_results]
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error(f"Quota exceeded for niche search: {niche}")
            return []
        logger.error(f"Error searching channels: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching channels: {e}")
        return []

def get_channel_details(channel_id):
    if not youtube:
        logger.error("YouTube API client not initialized")
        return None
    try:
        request = youtube.channels().list(
            part="snippet,statistics",
            id=channel_id
        )
        response = request.execute()
        if "items" not in response or not response["items"]:
            logger.warning(f"No channel data found for channel_id: {channel_id}")
            return None
        channel = response["items"][0]
        profile_picture = channel["snippet"]["thumbnails"]["default"]["url"]
        channel_url = f"https://www.youtube.com/channel/{channel_id}"
        return {
            "name": channel["snippet"]["title"],
            "description": channel["snippet"]["description"],
            "subscribers": int(channel["statistics"].get("subscriberCount", 0)),
            "total_views": int(channel["statistics"].get("viewCount", 0)),
            "video_count": int(channel["statistics"].get("videoCount", 0)),
            "profile_picture": profile_picture,
            "channel_url": channel_url
        }
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error(f"Quota exceeded for channel_id: {channel_id}")
            return None
        logger.error(f"Error fetching details for {channel_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching details for {channel_id}: {e}")
        return None

def get_recent_videos(channel_id, max_results=10, niche_keywords=None):
    if not youtube:
        logger.error("YouTube API client not initialized")
        return []
    try:
        search_request = youtube.search().list(
            part="id,snippet",
            channelId=channel_id,
            maxResults=max_results,
            order="date"
        )
        search_response = search_request.execute()
        if "items" not in search_response or not search_response["items"]:
            logger.warning(f"No recent videos found for channel_id: {channel_id}")
            return []
        video_ids = [item["id"]["videoId"] for item in search_response["items"] if item["id"].get("kind") == "youtube#video"]
        if not video_ids:
            logger.warning(f"No videos found for channel_id: {channel_id}")
            return []
        video_titles = [item["snippet"]["title"] for item in search_response["items"] if item["id"].get("kind") == "youtube#video"]
        video_descriptions = [item["snippet"]["description"] for item in search_response["items"] if item["id"].get("kind") == "youtube#video"]
        video_dates = [item["snippet"]["publishedAt"] for item in search_response["items"] if item["id"].get("kind") == "youtube#video"]
        thumbnails = [item["snippet"]["thumbnails"]["default"]["url"] for item in search_response["items"] if item["id"].get("kind") == "youtube#video"]
        video_request = youtube.videos().list(
            part="statistics",
            id=",".join(video_ids)
        )
        video_response = video_request.execute()
        video_data = []
        for i, video in enumerate(video_response["items"]):
            stats = video["statistics"]
            text = (video_titles[i] + " " + video_descriptions[i])
            sentiment = analyze_sentiment_with_distilbert(text)
            sentiment_score = sentiment["score"]
            sentiment_label = sentiment["label"]
            keyword_match = 0
            if niche_keywords:
                for keyword in niche_keywords:
                    if re.search(keyword, text, re.IGNORECASE):
                        keyword_match = 1
                        break
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            engagement_rate = ((likes + comments) / views * 100) if views > 0 else 0
            video_data.append({
                "title": video_titles[i],
                "description": video_descriptions[i],
                "published_at": video_dates[i],
                "views": views,
                "likes": likes,
                "comments": comments,
                "engagement_rate": engagement_rate,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "keyword_match": keyword_match * 100,
                "text_content": text,
                "thumbnail": thumbnails[i]
            })
        logger.info(f"Fetched {len(video_data)} recent videos for channel_id: {channel_id}")
        return video_data
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error(f"Quota exceeded for channel_id: {channel_id}")
            return []
        logger.error(f"Error fetching videos for {channel_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching videos for {channel_id}: {e}")
        return []

# --- DistilBERT Functions ---
def analyze_sentiment_with_distilbert(text):
    if not sentiment_analyzer:
        logger.error("Sentiment analyzer not initialized")
        return {"score": 0, "label": "NEUTRAL"}
    try:
        text = text[:512]  # Truncate to avoid token limit issues
        result = sentiment_analyzer(text)[0]
        label = result["label"]
        score = result["score"]
        if label == "POSITIVE":
            sentiment_score = score
            sentiment_label = "POSITIVE"
        else:
            sentiment_score = -score
            sentiment_label = "NEGATIVE"
        if abs(sentiment_score) < 0.1:
            sentiment_label = "NEUTRAL"
        elif 0.1 <= abs(sentiment_score) < 0.5:
            sentiment_label = "MIXED"
        return {"score": sentiment_score, "label": sentiment_label}
    except Exception as e:
        logger.error(f"Error with DistilBERT sentiment analysis: {e}")
        return {"score": 0, "label": "NEUTRAL"}

def get_word_importance(text):
    if not tokenizer or not distilbert_model:
        logger.error("DistilBERT tokenizer or model not initialized")
        return {}
    try:
        inputs = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=512)
        words = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            embeddings = outputs.last_hidden_state[0][1:-1]
        importance_scores = torch.norm(embeddings, dim=1).numpy()
        word_scores = {word: float(score) for word, score in zip(words, importance_scores) if word not in STOPWORDS and len(word) > 3 and word not in ["[CLS]", "[SEP]"] and not word.startswith("##")}
        return word_scores
    except Exception as e:
        logger.error(f"Error calculating word importance: {e}")
        return {}

def extract_niche_keywords(all_videos):
    if not all_videos:
        logger.warning("No videos provided for niche keyword extraction")
        return []
    all_videos_flat = []
    for channel_videos in all_videos:
        all_videos_flat.extend(channel_videos)
    if not all_videos_flat:
        logger.warning("No videos available for keyword extraction")
        return []
    all_text = " ".join(video["text_content"] for video in all_videos_flat).lower()
    words = all_text.translate(str.maketrans("", "", string.punctuation)).split()
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 3]
    word_counts = Counter(filtered_words)
    top_words = [word for word, count in word_counts.most_common(20)]
    word_scores = get_word_importance(all_text)
    keyword_scores = {}
    for word in top_words:
        importance = word_scores.get(word, 0)
        frequency = word_counts[word]
        combined_score = (importance * 0.4) + (frequency / len(filtered_words) * 100 * 0.6)
        keyword_scores[word] = combined_score
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [keyword for keyword, _ in sorted_keywords[:5]]
    logger.info(f"Extracted niche keywords: {top_keywords}")
    return top_keywords

# --- Analysis Functions ---
def predict_future_performance(videos):
    try:
        if not videos:
            return {"predicted_engagement_rate": 0, "recommendation": "No videos available for prediction."}
        engagement_rates = [v["engagement_rate"] for v in videos]
        avg_engagement = sum(engagement_rates) / len(engagement_rates)
        mid = len(videos) // 2
        early_engagement = sum(v["engagement_rate"] for v in videos[mid:]) / (len(videos) - mid) if len(videos) > mid else 0
        late_engagement = sum(v["engagement_rate"] for v in videos[:mid]) / mid if mid > 0 else 0
        trend = (late_engagement - early_engagement) / early_engagement if early_engagement > 0 else 0
        predicted_engagement = avg_engagement * (1 + trend * 0.5)
        predicted_engagement = max(0, predicted_engagement)
        if trend > 0:
            recommendation = "Continue focusing on recent content themes, as engagement is trending upward."
        elif trend < 0:
            recommendation = "Consider revisiting themes from earlier videos, as engagement is trending downward."
        else:
            recommendation = "Engagement is stable; experiment with new content themes to boost growth."
        return {
            "predicted_engagement_rate": round(predicted_engagement, 2),
            "recommendation": recommendation
        }
    except Exception as e:
        logger.error(f"Error with heuristic prediction: {e}")
        return {"predicted_engagement_rate": 0, "recommendation": "Unable to predict due to an error."}

def explosions_popular_keywords(videos):
    if not videos:
        return {
            "popular_keywords": [],
            "keyword_analysis": "No videos available for keyword analysis."
        }
    all_text = " ".join(video["text_content"] for video in videos).lower()
    words = all_text.translate(str.maketrans("", "", string.punctuation)).split()
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 3]
    word_counts = Counter(filtered_words)
    top_keywords = [word for word, count in word_counts.most_common(10)]
    keyword_engagement = {}
    for keyword in top_keywords:
        total_engagement = 0
        video_count = 0
        for video in videos:
            if keyword in video["text_content"].lower():
                total_engagement += video["engagement_rate"]
                video_count += 1
        avg_engagement = total_engagement / video_count if video_count > 0 else 0
        keyword_engagement[keyword] = avg_engagement
    keyword_importance = {}
    for keyword in top_keywords:
        word_scores = get_word_importance(all_text)
        importance = word_scores.get(keyword, 0)
        combined_score = (keyword_engagement[keyword] * 0.7) + (importance * 0.3)
        keyword_importance[keyword] = combined_score
    sorted_keywords = sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)
    top_growth_keywords = sorted_keywords[:5]
    popular_keywords = [keyword for keyword, _ in top_growth_keywords[:3]]
    analysis = (
        f"Keywords like '{popular_keywords[0]}' and '{popular_keywords[1]}' are driving engagement due to their high frequency and relevance to the channel's content. "
        f"They appear in high-engagement videos and align with niche trends." if len(popular_keywords) >= 2 else "Insufficient keywords for detailed analysis."
    )
    return {
        "popular_keywords": popular_keywords,
        "keyword_analysis": analysis
    }

def extract_top_viral_keywords(all_videos):
    if not all_videos:
        return {
            "top_viral_keywords": [],
            "viral_keyword_analysis": "No videos available for viral keyword analysis.",
            "keyword_counts": {}
        }
    all_videos_flat = []
    for channel_videos in all_videos:
        all_videos_flat.extend(channel_videos)
    if not all_videos_flat:
        return {
            "top_viral_keywords": [],
            "viral_keyword_analysis": "No videos available for viral keyword analysis.",
            "keyword_counts": {}
        }
    all_videos_flat.sort(key=lambda x: x["views"], reverse=True)
    top_10_percent = max(1, int(len(all_videos_flat) * 0.1))
    viral_videos = all_videos_flat[:top_10_percent]
    viral_text = " ".join(video["text_content"] for video in viral_videos).lower()
    words = viral_text.translate(str.maketrans("", "", string.punctuation)).split()
    filtered_words = [word for word in words if word not in STOPWORDS and len(word) > 3]
    word_counts = Counter(filtered_words)
    top_keywords = [word for word, count in word_counts.most_common(10)]
    keyword_views = {}
    for keyword in top_keywords:
        total_views = 0
        video_count = 0
        for video in viral_videos:
            if keyword in video["text_content"].lower():
                total_views += video["views"]
                video_count += 1
        avg_views = total_views / video_count if video_count > 0 else 0
        keyword_views[keyword] = avg_views
    keyword_importance = {}
    keyword_counts = {}
    for keyword in top_keywords:
        word_scores = get_word_importance(viral_text)
        importance = word_scores.get(keyword, 0)
        combined_score = (keyword_views[keyword] * 0.7) + (importance * 0.3)
        keyword_importance[keyword] = combined_score
        count = sum(1 for video in viral_videos if keyword in video["text_content"].lower())
        keyword_counts[keyword] = count
    sorted_keywords = sorted(keyword_importance.items(), key=lambda x: x[1], reverse=True)
    top_viral_keywords = sorted_keywords[:5]
    top_keywords_list = [keyword for keyword, _ in top_viral_keywords[:3]]
    analysis = (
        f"The top viral keywords '{top_keywords_list[0]}', '{top_keywords_list[1]}', and '{top_keywords_list[2]}' are driving virality across channels in this niche. "
        f"These keywords are associated with videos that have the highest views, indicating strong audience interest and engagement." if len(top_keywords_list) >= 3 else "Insufficient viral keywords for detailed analysis."
    )
    return {
        "top_viral_keywords": top_keywords_list,
        "viral_keyword_analysis": analysis,
        "keyword_counts": keyword_counts
    }

def calculate_brand_suitability_description(channel_data):
    try:
        brand_alignment = channel_data["brand_alignment"]
        performance_score = channel_data["performance_score"]
        engagement_rate = channel_data["engagement_rate"]
        keyword_score = channel_data["keyword_score"]
        subscribers = channel_data["subscribers"]
        alignment_score = brand_alignment["brand_alignment_score"]
        alignment_details = brand_alignment["alignment_details"]
        avg_sentiment = brand_alignment["average_sentiment"]
        keyword_match_percentage = brand_alignment["keyword_match_percentage"]
        description = f"{channel_data['name']} is an excellent fit for promoting sustainable fashion and ethical living due to its strong alignment with brand values. "
        description += f"With a brand alignment score of {alignment_score}%, {alignment_details.lower()} "
        description += f"The channel's content reflects a positive sentiment (average score: {avg_sentiment:.2f}) and a high keyword match rate ({keyword_match_percentage:.2f}% of videos mention niche-relevant terms). "
        description += f"Additionally, the channel has a performance score of {performance_score:.2f}, an engagement rate of {engagement_rate:.2f}%, and a loyal audience of {subscribers:,} subscribers, making it a powerful platform to amplify your brand's message effectively."
        return description
    except Exception as e:
        logger.error(f"Error generating brand suitability description: {e}")
        return "Unable to generate brand suitability description due to missing data."

# --- Brand Alignment Analysis ---
def calculate_brand_alignment_score(videos, niche_keywords):
    if not videos:
        return {
            "brand_alignment_score": 0,
            "alignment_details": "No videos available for analysis.",
            "average_sentiment": 0,
            "keyword_match_percentage": 0
        }
    total_videos = len(videos)
    total_sentiment = 0
    keyword_matches = 0
    for video in videos:
        total_sentiment += video["sentiment_score"]
        if video["keyword_match"] > 0:
            keyword_matches += 1
    average_sentiment = total_sentiment / total_videos if total_videos > 0 else 0
    keyword_match_percentage = (keyword_matches / total_videos) * 100 if total_videos > 0 else 0
    sentiment_component = ((average_sentiment + 1) / 2) * 50
    keyword_component = keyword_match_percentage * 0.5
    brand_alignment_score = sentiment_component + keyword_component
    if brand_alignment_score >= 80:
        alignment_details = f"Strong alignment with brand values: Positive sentiment and high match with niche keywords ({', '.join(niche_keywords)})."
    elif brand_alignment_score >= 60:
        alignment_details = f"Moderate alignment: Fairly positive sentiment and some matches with niche keywords ({', '.join(niche_keywords)})."
    elif brand_alignment_score >= 40:
        alignment_details = f"Weak alignment: Mixed or neutral sentiment with limited matches with niche keywords ({', '.join(niche_keywords)})."
    else:
        alignment_details = f"Poor alignment: Negative sentiment and/or low matches with niche keywords ({', '.join(niche_keywords)})."
    return {
        "brand_alignment_score": round(brand_alignment_score, 2),
        "alignment_details": alignment_details,
        "average_sentiment": average_sentiment,
        "keyword_match_percentage": keyword_match_percentage
    }

# --- AI-Enhanced Analysis Functions ---
def analyze_content(videos, niche_keywords):
    total_sentiment = 0
    keyword_matches = 0
    total_videos = len(videos)
    if total_videos == 0:
        return {"sentiment": 0, "keyword_score": 0, "consistency": 0}
    sentiments = []
    for video in videos:
        score = video["sentiment_score"]
        sentiments.append(score)
        total_sentiment += score
        text = video["title"] + " " + video["description"]
        for keyword in niche_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                keyword_matches += 1
                break
    avg_sentiment = total_sentiment / total_videos
    keyword_score = (keyword_matches / total_videos) * 100
    consistency = statistics.stdev(sentiments) if len(sentiments) > 1 else 0
    consistency = max(0, 1 - consistency) * 100
    return {"sentiment": avg_sentiment, "keyword_score": keyword_score, "consistency": consistency}

def calculate_engagement(video_data):
    total_views = sum(video["views"] for video in video_data)
    total_likes = sum(video["likes"] for video in video_data)
    total_comments = sum(video["comments"] for video in video_data)
    total_videos = len(video_data)
    if total_views == 0 or total_videos == 0:
        return {"engagement_rate": 0, "avg_comments": 0, "growth_ratio": 0}
    engagement_rate = ((total_likes + total_comments) / total_views) * 100
    avg_comments = total_comments / total_videos
    mid = total_videos // 2
    early_videos = video_data[mid:]
    late_videos = video_data[:mid]
    early_engagement = sum(v["likes"] + v["comments"] for v in early_videos) / sum(v["views"] for v in early_videos) if sum(v["views"] for v in early_videos) > 0 else 0
    late_engagement = sum(v["likes"] + v["comments"] for v in late_videos) / sum(v["views"] for v in late_videos) if sum(v["views"] for v in late_videos) > 0 else 0
    growth_ratio = (late_engagement - early_engagement) * 100
    return {"engagement_rate": engagement_rate, "avg_comments": avg_comments, "growth_ratio": growth_ratio}

def calculate_performance_score(content_analysis, engagement):
    weights = {
        "sentiment": 0.25,
        "keyword_score": 0.25,
        "engagement_rate": 0.20,
        "consistency": 0.20,
        "growth_ratio": 0.10
    }
    sentiment_score = ((content_analysis["sentiment"] + 1) / 2) * 100
    keyword_score = content_analysis["keyword_score"]
    engagement_score = min(engagement["engagement_rate"] * 10, 100)
    consistency_score = content_analysis["consistency"]
    growth_score = min(max(engagement["growth_ratio"] + 50, 0), 100)
    total_score = (
        sentiment_score * weights["sentiment"] +
        keyword_score * weights["keyword_score"] +
        engagement_score * weights["engagement_rate"] +
        consistency_score * weights["consistency"] +
        growth_score * weights["growth_ratio"]
    )
    return round(total_score, 2)

# --- Enhanced Channel Report Generation Using Generative AI Approach ---
def generate_channel_report(channel_info, video_data, performance_score, brand_alignment, keyword_data, top_hashtags, niche_keywords):
    """Generate a detailed textual report summarizing channel analytics using a generative AI-like approach."""
    try:
        # Calculate average metrics
        avg_engagement = sum(v["engagement_rate"] for v in video_data) / len(video_data) if video_data else 0
        total_views = sum(v["views"] for v in video_data) if video_data else 0
        avg_sentiment = brand_alignment["average_sentiment"]
        keyword_match_percentage = brand_alignment["keyword_match_percentage"]

        # Determine performance remarks
        performance_remark = "excellent" if performance_score >= 80 else "good" if performance_score >= 60 else "needs improvement"
        alignment_remark = "strong" if brand_alignment['brand_alignment_score'] >= 80 else "moderate" if brand_alignment['brand_alignment_score'] >= 60 else "weak"

        # Format lists for report
        hashtag_list = ", ".join([h['hashtag'] for h in top_hashtags[:3]]) if top_hashtags else "none"
        keyword_list = ", ".join(keyword_data['popular_keywords'][:3]) if keyword_data['popular_keywords'] else "none"
        niche_keyword_list = ", ".join(niche_keywords[:3]) if niche_keywords else "none"

        # Generate the report using a structured, generative AI-like approach
        report = f"📊 Channel Analytics Report for {channel_info['name']}\n\n"
        report += "--------------------------------------------------\n"
        report += "Overview\n"
        report += "--------------------------------------------------\n"
        report += f"{channel_info['name']} is a YouTube channel with {channel_info['subscribers']:,} subscribers, {channel_info['video_count']} videos, and a total of {channel_info['total_views']:,} views. "
        report += f"The channel focuses on themes related to {niche_keyword_list}, engaging its audience with content that aligns with these topics.\n\n"

        report += "--------------------------------------------------\n"
        report += "Performance Metrics\n"
        report += "--------------------------------------------------\n"
        report += f"- Performance Score: {performance_score:.2f}/100 ({performance_remark})\n"
        report += f"- Average Engagement Rate: {avg_engagement:.2f}%\n"
        report += f"- Total Views (Recent Videos): {total_views:,}\n"
        report += f"- Sentiment: {avg_sentiment:.2f} (on a scale of -1 to 1)\n"
        report += f"- Keyword Match Rate: {keyword_match_percentage:.2f}% of videos mention niche-relevant terms\n\n"

        report += "--------------------------------------------------\n"
        report += "Content Analysis\n"
        report += "--------------------------------------------------\n"
        report += f"Popular Hashtags: {hashtag_list}\n"
        report += f"Key Content Themes: {keyword_list}\n"
        report += f"The channel's content resonates well with its audience, as evidenced by the frequent use of hashtags like {hashtag_list} and themes such as {keyword_list}. "
        report += f"These elements align with the niche keywords ({niche_keyword_list}), ensuring relevance to the target audience.\n\n"

        report += "--------------------------------------------------\n"
        report += "Brand Alignment\n"
        report += "--------------------------------------------------\n"
        report += f"- Brand Alignment Score: {brand_alignment['brand_alignment_score']:.2f}/100 ({alignment_remark})\n"
        report += f"- Alignment Details: {brand_alignment['alignment_details']}\n"
        report += f"The channel's alignment with brand values like '{BRAND_VALUES}' is {alignment_remark.lower()}, making it a {'strong' if alignment_remark == 'strong' else 'potential'} candidate for brand collaborations.\n\n"

        report += "--------------------------------------------------\n"
        report += "Recommendations\n"
        report += "--------------------------------------------------\n"
        report += f"To enhance performance, {channel_info['name']} should continue leveraging popular hashtags ({hashtag_list}) and focus on content themes like {keyword_list}. "
        report += f"Additionally, maintaining consistency in sentiment and engagement can further strengthen audience loyalty and brand partnerships.\n"

        return report
    except Exception as e:
        logger.error(f"Error generating channel report: {e}")
        return "Unable to generate channel report due to missing data."

def analyze_influencer(channel_id, min_subscribers=0, max_subscribers=float('inf'), niche_keywords=None):
    channel_info = get_channel_details(channel_id)
    if not channel_info or channel_info["subscribers"] < min_subscribers or channel_info["subscribers"] > max_subscribers:
        logger.warning(f"Channel {channel_id} filtered out: {channel_info}")
        return None
    video_data = get_recent_videos(channel_id, niche_keywords=niche_keywords)
    if not video_data:
        video_data = []  # Ensure video_data is always a list
    content_analysis = analyze_content(video_data, niche_keywords or [])
    engagement = calculate_engagement(video_data)
    performance_score = calculate_performance_score(content_analysis, engagement)
    prediction = predict_future_performance(video_data)
    brand_alignment = calculate_brand_alignment_score(video_data, niche_keywords or [])
    keyword_data = explosions_popular_keywords(video_data)

    # Extract hashtags
    all_hashtags = []
    for video in video_data:
        text = video["title"] + " " + video["description"]
        all_hashtags.extend(extract_hashtags(text))
    top_hashtags = [{"hashtag": h, "count": c} for h, c in Counter(all_hashtags).most_common(5)]

    # Generate report
    report = generate_channel_report(channel_info, video_data, performance_score, brand_alignment, keyword_data, top_hashtags, niche_keywords or [])

    # Ensure all fields are present, even if empty
    influencer_data = {
        "name": channel_info["name"],
        "subscribers": channel_info["subscribers"],
        "total_views": channel_info["total_views"],
        "video_count": channel_info["video_count"],
        "engagement_rate": engagement["engagement_rate"],
        "avg_comments": engagement["avg_comments"],
        "keyword_score": content_analysis["keyword_score"],
        "performance_score": performance_score,
        "growth_ratio": engagement["growth_ratio"],
        "channel_id": channel_id,
        "profile_picture": channel_info["profile_picture"],
        "channel_url": channel_info["channel_url"],
        "videos": video_data,
        "prediction": prediction,
        "brand_alignment": brand_alignment,
        "popular_keywords": keyword_data["popular_keywords"],
        "keyword_analysis": keyword_data["keyword_analysis"],
        "brand_suitability": calculate_brand_suitability_description({
            "name": channel_info["name"],
            "brand_alignment": brand_alignment,
            "performance_score": performance_score,
            "engagement_rate": engagement["engagement_rate"],
            "keyword_score": content_analysis["keyword_score"],
            "subscribers": channel_info["subscribers"]
        }),
        "top_hashtags": top_hashtags,
        "report": report,
        "top_viral_keywords": [],  # Will be populated later
        "viral_keyword_analysis": "",
        "keyword_counts": {}
    }
    return influencer_data

def find_influencers_in_niche(niche, min_subscribers=0, max_subscribers=float('inf')):
    channels = search_channels_by_niche(niche)
    influencer_list = []
    all_videos = []
    for channel in channels:
        video_data = get_recent_videos(channel["channel_id"], niche_keywords=None)
        if video_data:
            all_videos.append(video_data)
    niche_keywords = extract_niche_keywords(all_videos)
    for channel in channels:
        influencer_data = analyze_influencer(channel["channel_id"], min_subscribers, max_subscribers, niche_keywords)
        if influencer_data:
            influencer_list.append(influencer_data)
    viral_keyword_data = extract_top_viral_keywords(all_videos)
    for influencer in influencer_list:
        influencer["top_viral_keywords"] = viral_keyword_data["top_viral_keywords"]
        influencer["viral_keyword_analysis"] = viral_keyword_data["viral_keyword_analysis"]
        influencer["keyword_counts"] = viral_keyword_data["keyword_counts"]
    influencer_list.sort(key=lambda x: x["performance_score"], reverse=True)
    logger.info(f"Returning {len(influencer_list)} influencers for niche: {niche}")
    return influencer_list

def extract_channel_id_from_url(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/channel\/([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/@([a-zA-Z0-9_-]+)',
        r'(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            channel_id = match.group(1)
            if pattern.endswith('@([a-zA-Z0-9_-]+)'):
                channel = get_channel_by_name(channel_id)
                if channel:
                    return channel["channel_id"]
            return channel_id
    logger.warning(f"Invalid YouTube channel URL: {url}")
    return None

def get_channel_by_name(channel_name):
    if not youtube:
        logger.error("YouTube API client not initialized")
        return None
    try:
        request = youtube.search().list(
            part="snippet",
            q=channel_name,
            type="channel",
            maxResults=1,
            relevanceLanguage="en"
        )
        response = request.execute()
        if "items" in response and response["items"]:
            channel = response["items"][0]
            return {
                "channel_id": channel["snippet"]["channelId"],
                "name": channel["snippet"]["title"],
                "description": channel["snippet"]["description"]
            }
        return None
    except googleapiclient.errors.HttpError as e:
        if e.resp.status == 403 and 'quotaExceeded' in str(e):
            logger.error(f"Quota exceeded for channel search: {channel_name}")
            return None
        logger.error(f"Error searching channel by name {channel_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error searching channel by name {channel_name}: {e}")
        return None

def lmao():
    print('Lol, this is a test function.')

if __name__ == "__main__":
    niche = "sustainable fashion"
    influencers = find_influencers_in_niche(niche, min_subscribers=1000)
    print(json.dumps(influencers, indent=2))
