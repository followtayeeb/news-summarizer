import requests
import json
from bs4 import BeautifulSoup
from notion_client import Client
import os
from dotenv import load_dotenv
import logging
from datetime import date
import time
import random

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
NOTION_PAGE_ID = os.getenv('NOTION_PAGE_ID')

INDUSTRIES = {
    'SEO': '"search engine optimization" OR "SEO strategies" OR "organic search"',
    'Digital Marketing': '"digital marketing trends" OR "online advertising" OR "content marketing"',
    'AI': '"artificial intelligence" OR "machine learning" OR "AI applications"'
}

def fetch_news(query):
    api_key = os.getenv('NEWS_API_KEY')
    url = f'https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}'
    response = requests.get(url)
    news = response.json()
    return news['articles'][:15]  # Fetch top 15 articles

def call_groq_with_retry(messages, max_retries=5, initial_delay=1):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 10000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"HTTP error occurred: {e}")
                raise
        except KeyError:
            logger.error("Unexpected response format from GROQ API")
            raise
    
    logger.error("Max retries reached. Unable to get a valid response from GROQ API.")
    raise Exception("Max retries reached for GROQ API call")

def check_relevance(article, industry):
    prompt = f"""Determine if the following article is relevant to {industry}.
    
Title: {article['title']}
Description: {article['description']}

Respond with only 'Relevant' or 'Not Relevant'.
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_groq_with_retry(messages)
    return response.strip().lower() == 'relevant'

def summarize_article(article):
    prompt = f"""Summarize the following article in a concise manner:

Title: {article['title']}
Description: {article['description']}
Content: {article['content']}

Provide a summary in the following JSON format:
{{
    "title": "Brief title",
    "summary": "Comprehensive summary",
    "key_points": ["Point 1", "Point 2", "Point 3", "Point 4"]
}}
"""
    messages = [{"role": "user", "content": prompt}]
    response = call_groq_with_retry(messages)
    summary = json.loads(response)
    summary['source'] = article['source']['name']
    summary['url'] = article['url']
    summary['image_url'] = article['urlToImage']
    return summary

def create_block(block_type, content):
    if block_type == "heading_2":
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        } 
    elif block_type == "heading_3":
        return {
            "object": "block",
            "type": "heading_3",
            "heading_3": {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        }
    elif block_type == "paragraph":
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        }
    elif block_type == "bulleted_list_item":
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        }
    elif block_type == "image":
        return {
            "object": "block",
            "type": "image",
            "image": {
                "type": "external",
                "external": {
                    "url": content
                }
            }
        }

def is_valid_image_url(url):
    if not url:
        return False
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('Content-Type', '')
        return response.status_code == 200 and content_type.startswith('image/')
    except requests.RequestException:
        return False

def update_notion(all_summaries):
    notion = Client(auth=NOTION_API_KEY)
    today = date.today().strftime("%Y-%m-%d")
    
    try:
        new_page = notion.pages.create(
            parent={"page_id": NOTION_PAGE_ID},
            properties={
                "title": {"title": [{"text": {"content": f"{today} - News Summary"}}]},
            },
        )
        logger.info(f"Created new page for {today}")
    except Exception as e:
        logger.error(f"Failed to create new page: {str(e)}")
        return

    for industry, summaries in all_summaries.items():
        blocks = [create_block("heading_2", industry)]
        
        for summary in summaries:
            blocks.extend([
                create_block("heading_3", summary['title']),
                create_block("paragraph", summary['summary']),
                create_block("bulleted_list_item", "Key Points:")
            ])
            
            for point in summary['key_points']:
                blocks.append(create_block("bulleted_list_item", point))
            
            if summary['image_url'] and is_valid_image_url(summary['image_url']):
                blocks.append(create_block("image", summary['image_url']))
            
            # Create a clickable link for the source
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": f"Source: {summary['source']} - "}
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": "Read full article",
                                "link": {"url": summary['url']}
                            }
                        }
                    ]
                }
            })
            blocks.append(create_block("paragraph", ""))  # Add space between summaries
        
        try:
            notion.blocks.children.append(new_page["id"], children=blocks)
            logger.info(f"Appended summaries for {industry}")
        except Exception as e:
            logger.error(f"Failed to append summaries for {industry}: {str(e)}")
            logger.debug(f"Blocks structure: {json.dumps(blocks, indent=2)}")

def process_industry(industry, query):
    logger.info(f"Processing news for {industry}")
    articles = fetch_news(query)
    logger.info(f"Fetched {len(articles)} articles for {industry}")
    
    relevant_articles = []
    for article in articles:
        try:
            if check_relevance(article, industry):
                relevant_articles.append(article)
                if len(relevant_articles) == 5:
                    break
        except Exception as e:
            logger.error(f"Error checking relevance for article in {industry}: {str(e)}")
    
    logger.info(f"Found {len(relevant_articles)} relevant articles for {industry}")
    
    summaries = []
    for article in relevant_articles:
        try:
            summaries.append(summarize_article(article))
        except Exception as e:
            logger.error(f"Error summarizing article in {industry}: {str(e)}")
    
    logger.info(f"Summarized {len(summaries)} articles for {industry}")
    
    return summaries

def main():
    all_summaries = {}
    for industry, query in INDUSTRIES.items():
        all_summaries[industry] = process_industry(industry, query)
    
    logger.info(f"Total industries processed: {len(all_summaries)}")
    for industry, summaries in all_summaries.items():
        logger.info(f"Industry: {industry}, Number of summaries: {len(summaries)}")
    
    update_notion(all_summaries)

if __name__ == "__main__":
    main()