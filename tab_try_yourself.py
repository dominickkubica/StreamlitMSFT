# tab_try_yourself.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import logging
import json
import openai
import os
import re
from tenacity import stop_after_attempt
from tenacity import wait_exponential
from tenacity import retry

# Setup simplified logging
log_output = io.StringIO()
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s', stream=log_output)
logger = logging.getLogger("earnings_analyzer")

# Recent Microsoft earnings dates
MSFT_EARNINGS_DATES = [
    "2023-01-24", "2023-04-25", "2023-07-25", "2023-10-24",
    "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-30",
    "2025-01-29", "2025-04-30"
]

# Hardcoded file paths
STOCK_DATA_PATH = "C:\\Users\\kubic\\OneDrive\\Desktop\\MSFTSTREAMLIT\\HistoricalData_1747025804532.csv"
SENTIMENT_DATA_PATH = "C:\\Users\\kubic\\OneDrive\\Desktop\\MSFTSTREAMLIT\\MSFTQ2_preload.csv.csv"

# -------------------------------------------------------------------------------
# Stock data related functions
# -------------------------------------------------------------------------------

# Custom metric function with direct HTML color control
def colored_metric(label, value, delta, suffix="%", prefix="", vs_text=""):
    # Determine color and arrow based on delta value
    if delta < 0:
        color = "#D13438"  # Microsoft red
        arrow = "↓"
    else:
        color = "#107C10"  # Microsoft green
        arrow = "↑"
    
    # Format the delta with the colored arrow
    vs_part = f" {vs_text}" if vs_text else ""
    delta_html = f'<span style="color:{color}">{arrow} {abs(delta):.2f}{suffix}{vs_part}</span>'
    
    # Create HTML for the metric
    html = f"""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;">{prefix}{value}</div>
        <div style="font-size: 0.9rem;">{delta_html}</div>
    </div>
    """
    
    # Display using markdown with HTML
    st.markdown(html, unsafe_allow_html=True)

# Function to load stock data from CSV
def load_stock_data(file_path):
    """Load stock data from CSV and perform minimal processing"""
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
        
    try:
        # Load raw data
        df = pd.read_csv(file_path)
        
        # Identify date and price columns
        date_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower:
                date_col = col
            elif 'close' in col_lower or 'last' in col_lower:
                price_col = col
        
        if not date_col or not price_col:
            st.error("Could not identify date and price columns")
            return None
        
        # Convert columns to proper types
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Find OHLC columns if they exist
        ohlc_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'open' in col_lower:
                ohlc_cols['open'] = col
            elif 'high' in col_lower:
                ohlc_cols['high'] = col
            elif 'low' in col_lower:
                ohlc_cols['low'] = col
            elif 'volume' in col_lower:
                ohlc_cols['volume'] = col
        
        # Convert OHLC columns to numeric if they exist
        for key, col in ohlc_cols.items():
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Drop any rows with invalid dates or prices
        df = df.dropna(subset=[date_col, price_col])
        
        # Sort by date (oldest first)
        df = df.sort_values(by=date_col)
        
        return df, date_col, price_col, ohlc_cols
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Function to find closest trading day to a given date
def find_closest_trading_day(df, date_col, target_date, direction='both'):
    """
    Find the closest trading day to the target date
    direction: 'both' (default), 'before', or 'after'
    """
    if direction == 'before':
        # Only consider dates before or on the target date
        valid_dates = [d for d in df[date_col].dt.date if d <= target_date]
        if not valid_dates:
            return None
        return max(valid_dates)
    
    elif direction == 'after':
        # Only consider dates after the target date
        valid_dates = [d for d in df[date_col].dt.date if d >= target_date]
        if not valid_dates:
            return None
        return min(valid_dates)
    
    else:  # both
        # Find the closest date in either direction
        min_diff = timedelta(days=365)
        closest_date = None
        
        for date in df[date_col].dt.date:
            diff = abs(date - target_date)
            if diff < min_diff:
                min_diff = diff
                closest_date = date
        
        return closest_date

# Generate sample stock data if actual data is not available
def generate_sample_stock_data(start_date, end_date):
    """Generate sample stock data if actual data is not available"""
    try:
        # Generate sample data
        date_range = pd.date_range(start=start_date, end=end_date)
        sample_data = pd.DataFrame(index=date_range)
        
        # Set column names
        sample_data['Date'] = date_range
        sample_data['Open'] = np.linspace(250, 280, len(date_range)) + np.random.normal(0, 5, len(date_range))
        sample_data['High'] = sample_data['Open'] + np.random.uniform(1, 10, len(date_range))
        sample_data['Low'] = sample_data['Open'] - np.random.uniform(1, 10, len(date_range))
        sample_data['Close'] = sample_data['Open'] + np.random.normal(0, 5, len(date_range))
        sample_data['Volume'] = np.random.uniform(20000000, 50000000, len(date_range))
        
        # Create the same structure as load_stock_data
        date_col = 'Date'
        price_col = 'Close'
        ohlc_cols = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }
        
        return sample_data, date_col, price_col, ohlc_cols
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return None

# Creates earnings impact analysis visualization
def create_earnings_impact_analysis(stock_data, date_col, price_col, ohlc_cols, highlight_date):
    """Create detailed analysis of stock movement around earnings date"""
    # Find closest trading day to the earnings date
    trading_day = find_closest_trading_day(stock_data, date_col, highlight_date)
    
    if not trading_day:
        return st.warning(f"No trading data found for {highlight_date}")
    
    # Get price on earnings day
    earnings_day_data = stock_data[stock_data[date_col].dt.date == trading_day]
    if earnings_day_data.empty:
        return st.warning(f"No stock data available for {trading_day}")
    
    earnings_price = earnings_day_data[price_col].iloc[0]
    
    # Create detailed earnings impact section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("### Stock Price Impact")
        
        # Define periods to analyze
        periods = [1, 3, 5, 10, 20, 60]
        period_data = []
        
        # Calculate pre-earnings price changes
        for days in periods:
            # Find date before earnings
            pre_date = trading_day - timedelta(days=days)
            pre_trading_day = find_closest_trading_day(stock_data, date_col, pre_date, 'before')
            
            if pre_trading_day:
                pre_data = stock_data[stock_data[date_col].dt.date == pre_trading_day]
                if not pre_data.empty:
                    pre_price = pre_data[price_col].iloc[0]
                    pre_change = ((earnings_price - pre_price) / pre_price) * 100
                    period_data.append({
                        "Period": f"{days}D Before",
                        "Price": pre_price,
                        "Change": pre_change,
                        "Direction": "pre"
                    })
        
        # Calculate post-earnings price changes
        for days in periods:
            # Find date after earnings
            post_date = trading_day + timedelta(days=days)
            post_trading_day = find_closest_trading_day(stock_data, date_col, post_date, 'after')
            
            if post_trading_day:
                post_data = stock_data[stock_data[date_col].dt.date == post_trading_day]
                if not post_data.empty:
                    post_price = post_data[price_col].iloc[0]
                    post_change = ((post_price - earnings_price) / earnings_price) * 100
                    period_data.append({
                        "Period": f"{days}D After",
                        "Price": post_price,
                        "Change": post_change,
                        "Direction": "post"
                    })
        
        # Create a DataFrame for better display
        period_df = pd.DataFrame(period_data)
        
        # Group and format the data
        pre_df = period_df[period_df["Direction"] == "pre"].sort_values("Period")
        post_df = period_df[period_df["Direction"] == "post"].sort_values("Period")
        
        # Display the earnings day price using a simpler metric without delta
        st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Earnings Day Price</div>"
                   f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${earnings_price:.2f}</div>", 
                   unsafe_allow_html=True)
        
        # Create a better visualization of the pre/post earnings changes
        st.write("#### Price Changes Around Earnings")
        
        # Create 2 columns for pre and post data
        pre_col, post_col = st.columns(2)
        
        # Use custom colored_metric for pre-earnings
        with pre_col:
            st.write("Pre-Earnings Performance:")
            for _, row in pre_df.iterrows():
                period = row["Period"]
                price = row["Price"]
                change = row["Change"]
                
                # Use our custom HTML-based colored metric
                colored_metric(
                    period, 
                    f"${price:.2f}", 
                    change
                )
        
        # Use custom colored_metric for post-earnings
        with post_col:
            st.write("Post-Earnings Performance:")
            for _, row in post_df.iterrows():
                period = row["Period"]
                price = row["Price"]
                change = row["Change"]
                
                # Use our custom HTML-based colored metric
                colored_metric(
                    period, 
                    f"${price:.2f}", 
                    change
                )
        
        # Create a zoomed-in chart showing the period around earnings
        st.write("#### Zoom View Around Earnings Date")
        
        # Define range for zoom chart: 30 days before to 30 days after
        zoom_start = trading_day - timedelta(days=30)
        zoom_end = trading_day + timedelta(days=30)
        
        # Find closest trading days to these dates
        zoom_start_day = find_closest_trading_day(stock_data, date_col, zoom_start, 'before')
        zoom_end_day = find_closest_trading_day(stock_data, date_col, zoom_end, 'after')
        
        if zoom_start_day and zoom_end_day:
            # Filter data for the zoom range
            zoom_mask = (stock_data[date_col].dt.date >= zoom_start_day) & (stock_data[date_col].dt.date <= zoom_end_day)
            zoom_df = stock_data[zoom_mask].copy()
            
            # Create zoom chart
            zoom_fig = go.Figure()
            
            # Add price line
            zoom_fig.add_trace(go.Scatter(
                x=zoom_df[date_col],
                y=zoom_df[price_col],
                mode='lines',
                name='MSFT',
                line=dict(color='#0078D4', width=2)
            ))
            
            # Add earnings day vertical line
            zoom_fig.add_vline(
                x=trading_day,
                line=dict(color='#FF0000', width=2)
            )
            
            # Add earnings day annotation
            zoom_fig.add_annotation(
                x=trading_day,
                y=zoom_df[price_col].max() * 1.02,
                text="Earnings Call",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#FF0000',
                font=dict(color='#FF0000', size=12),
                align="center"
            )
            
            # Update layout
            zoom_fig.update_layout(
                title=f"MSFT Stock Price Around {trading_day.strftime('%Y-%m-%d')} Earnings",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400,
                template="plotly_white",
                hovermode="x unified"
            )
            
            # Display the zoom chart
            st.plotly_chart(zoom_fig, use_container_width=True, key="earnings_zoom_chart")
    
    with col2:
        st.write("### Key Metrics")
        
        # Opening Gap section (if available)
        if ohlc_cols and 'open' in ohlc_cols:
            earnings_open = earnings_day_data[ohlc_cols['open']].iloc[0]
            
            # Find previous trading day
            prev_date = trading_day - timedelta(days=1)
            prev_trading_day = find_closest_trading_day(stock_data, date_col, prev_date, 'before')
            
            if prev_trading_day:
                prev_data = stock_data[stock_data[date_col].dt.date == prev_trading_day]
                if not prev_data.empty:
                    prev_close = prev_data[price_col].iloc[0]
                    gap_pct = ((earnings_open - prev_close) / prev_close) * 100
                    
                    # Use custom colored_metric for Opening Gap
                    colored_metric(
                        "Opening Gap", 
                        f"${earnings_open:.2f}", 
                        gap_pct,
                        "% from prev close"
                    )
        
        # Historical Context section
        st.write("#### Historical Context")
        
        # MSFT earnings dates
        earnings_dates = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in MSFT_EARNINGS_DATES]
        
        # Create comparison of earnings reactions
        earnings_reactions = []
        
        # Find post-earnings reaction for relevant earnings dates
        for e_date in earnings_dates:
            e_trading_day = find_closest_trading_day(stock_data, date_col, e_date)
            
            if e_trading_day:
                e_data = stock_data[stock_data[date_col].dt.date == e_trading_day]
                if not e_data.empty:
                    e_price = e_data[price_col].iloc[0]
                    
                    # Find price 3 days after earnings
                    post_date = e_trading_day + timedelta(days=3)
                    post_trading_day = find_closest_trading_day(stock_data, date_col, post_date, 'after')
                    
                    if post_trading_day:
                        post_data = stock_data[stock_data[date_col].dt.date == post_trading_day]
                        if not post_data.empty:
                            post_price = post_data[price_col].iloc[0]
                            pct_change = ((post_price - e_price) / e_price) * 100
                            
                            earnings_reactions.append({
                                "Date": e_trading_day,
                                "Change": pct_change,
                                "IsSelected": e_trading_day == trading_day
                            })
        
        if earnings_reactions:
            # Create a DataFrame for visualization
            reaction_df = pd.DataFrame(earnings_reactions)
            
            # Average reaction
            avg_reaction = reaction_df["Change"].mean()
            
            # Get current reaction
            current_reaction = 0
            if not reaction_df[reaction_df["IsSelected"]].empty:
                current_reaction = reaction_df[reaction_df["IsSelected"]]["Change"].iloc[0]
            
            # Use custom colored_metric for 3-Day Reaction
            colored_metric(
                "3-Day Reaction", 
                f"{current_reaction:.2f}%", 
                current_reaction - avg_reaction,
                "% vs avg"
            )
            
            # Create a comparison chart
            reaction_fig = go.Figure()
            
            # Add bars for all reactions
            for _, row in reaction_df.iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d")
                change = row["Change"]
                is_selected = row["IsSelected"]
                
                # Color bars based on price movement (up is green, down is red)
                bar_color = '#107C10' if change >= 0 else '#D13438'  # Green for price up, red for price down
                
                # Highlight selected date
                if is_selected:
                    bar_color = '#FF0000'  # Highlight in red
                
                reaction_fig.add_trace(go.Bar(
                    x=[date_str],
                    y=[change],
                    name=date_str,
                    marker_color=bar_color,
                    width=0.7
                ))
            
            # Add a line for the average
            reaction_fig.add_shape(
                type="line",
                x0=-0.5,
                y0=avg_reaction,
                x1=len(reaction_df) - 0.5,
                y1=avg_reaction,
                line=dict(color="black", width=2, dash="dash")
            )
            
            # Add annotation for the average
            reaction_fig.add_annotation(
                x=len(reaction_df) - 1,
                y=avg_reaction,
                text=f"Avg: {avg_reaction:.2f}%",
                showarrow=False,
                xanchor="right",
                font=dict(color="black", size=10)
            )
            
            # Update layout
            reaction_fig.update_layout(
                title="3-Day Reactions to Earnings",
                xaxis_title="Earnings Date",
                yaxis_title="% Change",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            
            # Display the chart
            st.plotly_chart(reaction_fig, use_container_width=True, key="earnings_reaction_chart")

# -------------------------------------------------------------------------------
# Sentiment Analysis related functions
# -------------------------------------------------------------------------------

# OpenAI API call with retries
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_openai_api(system_prompt, user_prompt, model="gpt-4-turbo", temperature=0.2):
    api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get('openai_api_key')
    if not api_key:
        st.error("OpenAI API key not found")
        return None
    
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        st.error(f"OpenAI API error: {e}")
        return None

# Load pre-categorized sentences from CSV
@st.cache_data
def load_precategorized_sentences(csv_path):
    """Load sentences already categorized by business line from CSV file"""
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Rename columns if needed
            if 'Sentence' in df.columns and 'BusinessUnit' in df.columns:
                df = df.rename(columns={'Sentence': 'text', 'BusinessUnit': 'business_line'})
            
            # Add needed columns if they don't exist
            if 'sentiment' not in df.columns:
                df['sentiment'] = None
            
            if 'section' not in df.columns:
                # Determine if there's a Q&A section based on content
                df['section'] = 'main'
                qa_indicators = ['Q:', 'Question:', 'Analyst:', 'Q&A']
                
                for indicator in qa_indicators:
                    mask = df['text'].str.contains(indicator, case=False, na=False)
                    if mask.sum() > 3:  # If multiple sentences contain Q&A indicators
                        df.loc[mask, 'section'] = 'qa'
            
            if 'processed' not in df.columns:
                df['processed'] = False
                
            # Add paragraph info if not present
            if 'paragraph_id' not in df.columns:
                df['paragraph_id'] = range(len(df))
                df['paragraph'] = df['text'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
            
            # Make sure we don't have NaN values in important columns
            df = df.fillna({'text': '', 'business_line': 'Other', 'section': 'main'})
            
            logger.info(f"Loaded {len(df)} pre-categorized sentences from CSV")
            return df
        else:
            st.warning(f"File not found: {csv_path}, using sample data instead")
            # Create sample sentences for demonstration
            sample_sentences = [
                {"text": "Our cloud revenue grew 28% year-over-year.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Office 365 continues to perform strongly with high adoption rates.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Gaming revenue declined by 5% due to challenging comps.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Azure growth was 27%, which slightly underperformed our guidance.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Q: What are you seeing in terms of AI adoption among enterprise customers?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We're seeing tremendous interest in our AI solutions across all segments.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Our commercial cloud revenue was $29.6 billion, up 23% year-over-year.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "LinkedIn revenue increased 10% with continued strength in Marketing Solutions.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Windows OEM revenue decreased 2% as the PC market continues to stabilize.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Our Dynamics products and cloud services revenue increased 18%.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
                {"text": "Q: How do you see the competitive landscape evolving in AI?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We believe our comprehensive AI approach gives us a significant advantage.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Server products and cloud services revenue increased 25%, driven by Azure.", "business_line": "Intelligent Cloud", "sentiment": None, "section": "main"},
                {"text": "Operating expenses increased 11% reflecting investments in AI and cloud.", "business_line": "Other", "sentiment": None, "section": "main"},
                {"text": "Search and news advertising revenue increased 18% driven by higher search volume.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "We returned $8.8 billion to shareholders through dividends and share repurchases.", "business_line": "Other", "sentiment": None, "section": "main"},
                {"text": "Q: Can you provide more color on your AI infrastructure investments?", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "A: We continue to expand our data center footprint to support AI workloads.", "business_line": "Q&A Section", "sentiment": None, "section": "qa"},
                {"text": "Surface revenue decreased 8% due to the challenging device market.", "business_line": "More Personal Computing", "sentiment": None, "section": "main"},
                {"text": "Office 365 Commercial revenue grew 15% driven by seat growth and ARPU increase.", "business_line": "Productivity and Business Processes", "sentiment": None, "section": "main"},
            ]
            df = pd.DataFrame(sample_sentences)
            df['processed'] = False
            df['paragraph_id'] = range(len(df))
            df['paragraph'] = df['text'].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else str(x))
            return df
    except Exception as e:
        logger.error(f"Error loading pre-categorized sentences: {e}")
        st.error(f"Error loading pre-categorized sentences: {e}")
        return pd.DataFrame()

# Extract segment data from business lines
def extract_segment_data(sentences_df):
    """Extract segment info from business lines"""
    if sentences_df.empty or 'business_line' not in sentences_df.columns:
        return {
            "SegmentMap": {},
            "BusinessLines": [],
            "BusinessToSegment": {}
        }
    
    # Get unique business lines
    business_lines = sentences_df['business_line'].unique().tolist()
    
    # Microsoft's standard business segments
    segment_map = {
        "Productivity and Business Processes": ["Office Commercial", "Office Consumer", "LinkedIn", "Dynamics"],
        "Intelligent Cloud": ["Server products and cloud services", "Enterprise Services", "Azure"],
        "More Personal Computing": ["Windows", "Devices", "Gaming", "Search and news advertising"]
    }
    
    # Create business to segment mapping
    business_to_segment = {}
    for segment, businesses in segment_map.items():
        for business in businesses:
            business_to_segment[business] = segment
    
    # Map any business lines not in the standard mapping to "Other"
    for business in business_lines:
        if business not in business_to_segment and business not in ["Q&A Section", "Other"]:
            # Handle the case where the business line might be the segment itself
            if business in segment_map:
                business_to_segment[business] = business
            else:
                business_to_segment[business] = "Other"
    
    # Add segment column to DataFrame if not present
    if 'segment' not in sentences_df.columns:
        sentences_df['segment'] = sentences_df['business_line'].map(
            lambda x: business_to_segment.get(x, "Other")
        )
    
    # Make sure Q&A and Other are included
    if "Q&A Section" not in business_lines:
        business_lines.append("Q&A Section")
    
    if "Other" not in business_lines:
        business_lines.append("Other")
    
    return {
        "SegmentMap": segment_map,
        "BusinessLines": business_lines,
        "BusinessToSegment": business_to_segment
    }

# Analyze sentiment for the pre-categorized sentences
def analyze_sentiment_for_sentences(sentences_df, progress_callback=None):
    """Analyze sentiment for pre-categorized sentences with batch processing"""
    if sentences_df.empty:
        return sentences_df
    
    # Create a copy to avoid warning about setting values on a slice
    sentences_df = sentences_df.copy()
    
    # Process in batches for efficiency
    batch_size = 50  # Smaller batch size to ensure proper analysis
    total_batches = (len(sentences_df) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        # Calculate batch start and end
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sentences_df))
        batch = sentences_df.iloc[start_idx:end_idx].copy()  # Create a copy to avoid warnings
        
        # Skip already processed sentences
        if 'processed' in batch.columns:
            batch_to_process = batch[~batch['processed']].copy()
            if batch_to_process.empty:
                if progress_callback:
                    progress_value = (batch_idx + 1) / total_batches
                    progress_callback(progress_value)
                continue
        else:
            batch_to_process = batch.copy()
        
        # Create batch text with context
        batch_texts = []
        for i, (idx, row) in enumerate(batch_to_process.iterrows()):
            batch_texts.append(f"Sentence {i+1}: {row['text']}")
        
        batch_text = "\n".join(batch_texts)
        
        # Create system prompt
        system_prompt = """
        You are an expert financial analyst specialized in Microsoft earnings call analysis.
        
        Determine the sentiment of each numbered sentence from an earnings call transcript.
        
        For sentiment, classify as:
        - "positive" (growth, success, exceeding expectations, improvements)
        - "neutral" (factual statements, on-target performance, stable metrics)
        - "negative" (declines, challenges, missed targets, concerns)
        
        Return ONLY a JSON array in this exact format:
        [
          {
            "sentence_num": 1,
            "sentiment": "positive/neutral/negative"
          },
          ... (entries for all sentences)
        ]
        """
        
        user_prompt = f"Analyze the sentiment of these earnings call statements:\n\n{batch_text}"
        
        # Call API
        response = call_openai_api(system_prompt, user_prompt)
        
        # Update progress
        if progress_callback:
            progress_value = (batch_idx + 1) / total_batches
            progress_callback(progress_value)
        
        if response:
            # Parse the response
            try:
                results = json.loads(response)
                
                # Update DataFrame with classifications
                for i, result in enumerate(results):
                    sentence_num = result.get('sentence_num')
                    if 1 <= sentence_num <= len(batch_to_process):
                        batch_idx_to_update = batch_to_process.index[sentence_num - 1]
                        sentences_df.at[batch_idx_to_update, 'sentiment'] = result.get('sentiment')
                        sentences_df.at[batch_idx_to_update, 'processed'] = True
                
                logger.info(f"Processed batch {batch_idx+1}/{total_batches} with {len(results)} classifications")
            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for batch {batch_idx+1}")
                # Try a simpler parsing approach if JSON fails
                try:
                    # Look for patterns like "1. Sentiment: positive"
                    lines = response.split('\n')
                    for i, line in enumerate(lines):
                        match = re.search(r'(\d+).*?sentiment:?\s*(\w+)', line.lower())
                        if match:
                            sentence_num = int(match.group(1))
                            sentiment = match.group(2).strip()
                            
                            if 1 <= sentence_num <= len(batch_to_process):
                                batch_idx_to_update = batch_to_process.index[sentence_num - 1]
                                
                                if sentiment in ['positive', 'neutral', 'negative']:
                                    sentences_df.at[batch_idx_to_update, 'sentiment'] = sentiment
                                else:
                                    sentences_df.at[batch_idx_to_update, 'sentiment'] = 'neutral'
                                
                                sentences_df.at[batch_idx_to_update, 'processed'] = True
                except Exception as e:
                    logger.error(f"Error in fallback parsing: {e}")
        else:
            logger.warning(f"No response for batch {batch_idx+1}")
            # Set default classifications for this batch
            for idx in batch_to_process.index:
                sentences_df.at[idx, 'sentiment'] = 'neutral'
                sentences_df.at[idx, 'processed'] = True
    
    # Fill in any remaining unprocessed sentences
    if 'processed' in sentences_df.columns and not sentences_df['processed'].all():
        unprocessed_mask = ~sentences_df['processed']
        sentences_df.loc[unprocessed_mask, 'sentiment'] = 'neutral'
        sentences_df.loc[unprocessed_mask, 'processed'] = True
    
    return sentences_df

# Aggregate sentiment by business line
def aggregate_sentiment_by_business_line(sentences_df):
    """
    Aggregate sentiment data by business line for visualization
    """
    if sentences_df.empty or 'business_line' not in sentences_df.columns:
        return pd.DataFrame()
    
    # Create a summary DataFrame
    aggregated = sentences_df.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
    
    # Calculate additional metrics
    aggregated['total'] = aggregated.sum(axis=1)
    if 'positive' in aggregated.columns and 'negative' in aggregated.columns:
        aggregated['net_sentiment'] = (aggregated['positive'] - aggregated['negative']) / aggregated['total']
    else:
        aggregated['net_sentiment'] = 0
    
    return aggregated

# Create sentiment timeline data for stock correlation
def create_sentiment_timeline(sentences_df, stock_data, date_col):
    """
    Create timeline data for sentiment correlation with stock price
    """
    if sentences_df.empty or stock_data is None or stock_data.empty:
        return pd.DataFrame()
    
    # Get dates from stock data
    dates = stock_data[date_col].dt.date.sort_values().unique()
    if len(dates) < 2:
        return pd.DataFrame()
    
    # Calculate sentiment scores by business line
    bl_sentiment = aggregate_sentiment_by_business_line(sentences_df)
    
    # Distribute sentiment scores across timeline
    timeline_data = []
    
    # Get 3-5 key dates spread across the stock data
    num_points = min(5, len(dates))
    date_indices = [int(i * (len(dates) - 1) / (num_points - 1)) for i in range(num_points)]
    key_dates = [dates[i] for i in date_indices]
    
    # Get business lines sorted by net sentiment
    business_lines = []
    if not bl_sentiment.empty:
        business_lines = bl_sentiment.sort_values('net_sentiment', ascending=False).index.tolist()
    
    # Assign each business line to different dates for visualization
    for i, business_line in enumerate(business_lines):
        if i >= len(key_dates):
            break
            
        date = key_dates[i]
        sentiment_value = bl_sentiment.loc[business_line, 'net_sentiment'] if business_line in bl_sentiment.index else 0
        
        # Map net sentiment to category
        if sentiment_value > 0.2:
            sentiment_category = 'positive'
        elif sentiment_value < -0.2:
            sentiment_category = 'negative'
        else:
            sentiment_category = 'neutral'
            
        timeline_data.append({
            'date': date,
            'business_line': business_line,
            'sentiment': sentiment_category,
            'sentiment_score': sentiment_value
        })
    
    return pd.DataFrame(timeline_data)

# Plot functions
def plot_sentiment_distribution(sentiment_data, microsoft_colors, title=None):
    """Create pie chart of sentiment distribution"""
    try:
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        
        # Create pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': microsoft_colors['success'],
                'neutral': microsoft_colors['accent'],
                'negative': microsoft_colors['danger']
            },
            hole=0.4
        )
        
        # Update layout with proper title
        chart_title = title if title else "Sentiment Distribution"
        fig.update_layout(
            title=chart_title,
            legend=dict(orientation="h"),
            title_font=dict(size=20),
            template="plotly_white"
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating sentiment distribution chart: {e}")
        return None

def plot_sentiment_by_business_line(sentiment_data, microsoft_colors):
    """Create stacked bar chart of sentiment by business line"""
    try:
        # Calculate sentiment by business segment
        segment_sentiment = sentiment_data.groupby('business_line')['sentiment'].value_counts().unstack().fillna(0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in segment_sentiment.columns:
                fig.add_trace(go.Bar(
                    x=segment_sentiment.index,
                    y=segment_sentiment[sentiment],
                    name=sentiment,
                    marker_color={
                        'positive': microsoft_colors['success'],
                        'neutral': microsoft_colors['accent'],
                        'negative': microsoft_colors['danger']
                    }[sentiment]
                ))
        
        # Update layout
        fig.update_layout(
            barmode='stack',
            legend=dict(orientation="h"),
            title="Sentiment by Business Segment",
            title_font=dict(size=20),
            template="plotly_white",
            xaxis_title="Business Segment",
            yaxis_title="Count",
            # Improve readability with rotated labels if many segments
            xaxis=dict(
                tickangle=-45 if len(segment_sentiment.index) > 4 else 0
            )
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating business segment chart: {e}")
        return None

def plot_sentiment_and_stock(sentiment_timeline, stock_data, date_col, price_col):
    """Create combined chart of sentiment and stock price"""
    if stock_data is None or sentiment_timeline.empty:
        return None
    
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add stock price line
        fig.add_trace(
            go.Scatter(
                x=stock_data[date_col], 
                y=stock_data[price_col],
                name='Stock Price',
                line=dict(color='#0078d4')
            ),
            secondary_y=True
        )
        
        # Add sentiment scores
        # Convert sentiment timeline date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(sentiment_timeline['date']):
            sentiment_timeline['date'] = pd.to_datetime(sentiment_timeline['date'])
        
        # Map categorical sentiment to numeric values for plotting
        sentiment_timeline['sentiment_value'] = sentiment_timeline['sentiment'].map({
            'positive': 1, 
            'neutral': 0, 
            'negative': -1
        })
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_timeline['date'],
                y=sentiment_timeline['sentiment_value'],
                name='Sentiment',
                mode='lines+markers',
                text=sentiment_timeline['business_line'],
                marker=dict(
                    size=10,
                    color=[
                        {
                            'positive': '#107c10',  # green
                            'neutral': '#5c2d91',   # purple
                            'negative': '#d13438'   # red
                        }[s] for s in sentiment_timeline['sentiment']
                    ]
                ),
                line=dict(color='#107c10')
            ),
            secondary_y=False
        )
        
        # Update layout
        title = "Sentiment vs Stock Price"
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(
            title_text="Sentiment Score (-1 to 1)",
            secondary_y=False,
            range=[-1.2, 1.2]
        )
        fig.update_yaxes(title_text="Stock Price ($)", secondary_y=True)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating sentiment and stock chart: {e}")
        return None

# Function to analyze chat messages with OpenAI
def analyze_chat_query(query, sentiment_data=None, stock_data=None, date_col=None, price_col=None):
    """Analyze a chat query using OpenAI API"""
    # Check if we have the necessary data
    has_sentiment = sentiment_data is not None and not sentiment_data.empty
    has_stock = stock_data is not None and not stock_data.empty and date_col is not None and price_col is not None
    
    # Prepare data summary for context
    context = ""
    
    # Include sentiment data summary if available
    if has_sentiment:
        sentiment_counts = sentiment_data['sentiment'].value_counts()
        total = len(sentiment_data)
        pos_pct = (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0
        neu_pct = (sentiment_counts.get('neutral', 0) / total * 100) if total > 0 else 0
        neg_pct = (sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0
        
        # Business line summary if available
        if 'business_line' in sentiment_data.columns:
            bl_sentiment = aggregate_sentiment_by_business_line(sentiment_data)
            bl_summary = []
            
            for bl, row in bl_sentiment.iterrows():
                net_sentiment = row.get('net_sentiment', 0)
                sentiment_word = "positive" if net_sentiment > 0.2 else "negative" if net_sentiment < -0.2 else "neutral"
                bl_summary.append(f"{bl}: {sentiment_word} ({net_sentiment:.2f})")
            
            context += "Business lines sentiment summary:\n" + "\n".join(bl_summary) + "\n\n"
        
        context += f"""
Sentiment Summary:
- Positive: {sentiment_counts.get('positive', 0)} statements ({pos_pct:.1f}%)
- Neutral: {sentiment_counts.get('neutral', 0)} statements ({neu_pct:.1f}%)
- Negative: {sentiment_counts.get('negative', 0)} statements ({neg_pct:.1f}%)
"""
    
    # Include stock data summary if available
    if has_stock:
        start_price = stock_data[price_col].iloc[0] if not stock_data.empty else 0
        end_price = stock_data[price_col].iloc[-1] if not stock_data.empty else 0
        percent_change = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
        min_price = stock_data[price_col].min() if not stock_data.empty else 0
        max_price = stock_data[price_col].max() if not stock_data.empty else 0
        
        # Date range
        start_date = stock_data[date_col].min().strftime('%Y-%m-%d') if not stock_data.empty else 'unknown'
        end_date = stock_data[date_col].max().strftime('%Y-%m-%d') if not stock_data.empty else 'unknown'
        
        context += f"""
Stock Price Summary ({start_date} to {end_date}):
- Starting Price: ${start_price:.2f}
- Ending Price: ${end_price:.2f} ({percent_change:.2f}%)
- Min Price: ${min_price:.2f}
- Max Price: ${max_price:.2f}
"""
    
    # System prompt for the chat
    system_prompt = f"""
You are a financial analyst assistant specialized in Microsoft earnings analysis.
You analyze sentiment data from earnings calls and stock price movements.
Answer questions clearly and concisely based on the data provided.

Here is the summary of the data you can reference:

{context}

When answering:
1. Be specific and data-driven, citing statistics when available
2. If information isn't available in the context, acknowledge this limitation
3. Keep answers direct and to the point
4. For complex queries, break down responses into clear sections
5. Use professional financial terminology but explain any technical concepts
"""
    
    # Call the API
    response = call_openai_api(system_prompt, query)
    return response

# Process workflow for analysis
def process_analysis_workflow(ticker_symbol, start_date, end_date):
    """Process the complete workflow for analysis"""
    try:
        with st.spinner("Running analysis..."):
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0.0)
            
            # Step 1: Load stock data
            status_placeholder.text("Loading stock data...")
            
            # Try to load from hardcoded path
            stock_data_result = load_stock_data(STOCK_DATA_PATH)
            
            # If loading failed, generate sample data
            if not stock_data_result:
                status_placeholder.text("Generating sample stock data...")
                stock_data_result = generate_sample_stock_data(start_date, end_date)
            
            # Update session state
            if stock_data_result:
                stock_data, date_col, price_col, ohlc_cols = stock_data_result
                st.session_state.stock_data = stock_data
                st.session_state.date_col = date_col
                st.session_state.price_col = price_col
                st.session_state.ohlc_cols = ohlc_cols
            else:
                st.error("Failed to load or generate stock data")
                return False
                
            progress_bar.progress(0.3)
            
            # Step 2: Load pre-categorized sentences
            status_placeholder.text("Loading pre-categorized sentences...")
            sentences_df = load_precategorized_sentences(SENTIMENT_DATA_PATH)
            progress_bar.progress(0.4)
            
            # Step 3: Extract segment info from the pre-categorized data
            status_placeholder.text("Extracting business structure...")
            segment_data = extract_segment_data(sentences_df)
            
            # Store in session state
            st.session_state.segment_data = segment_data
            st.session_state.business_lines = segment_data["BusinessLines"]
            progress_bar.progress(0.5)
            
            # Step 4: Analyze sentiment for each sentence
            status_placeholder.text("Analyzing sentiment...")
            
            # Function to update progress from within analyze_sentiment
            def update_progress(value):
                # Scale the progress from 0.5 to 0.8
                scaled_progress = 0.5 + (value * 0.3)
                progress_bar.progress(scaled_progress)
                status_placeholder.text(f"Analyzing sentiment: {int(value * 100)}% complete")
            
            # Run the sentiment analysis with progress updates
            classified_df = analyze_sentiment_for_sentences(sentences_df, update_progress)
            
            # Store complete results
            st.session_state.sentiment_results = classified_df
            
            # Check for Q&A section
            q_a_mask = classified_df['section'] == 'qa' if 'section' in classified_df.columns else pd.Series([False] * len(classified_df))
            st.session_state.has_qa_section = q_a_mask.any()
            
            # Extract Q&A results if applicable
            if st.session_state.has_qa_section:
                st.session_state.qa_results = classified_df[q_a_mask]
            
            # Step 5: Create timeline data for visualization
            status_placeholder.text("Creating visualization data...")
            sentiment_timeline = create_sentiment_timeline(classified_df, stock_data, date_col)
            st.session_state.sentiment_timeline = sentiment_timeline
            progress_bar.progress(1.0)
            
            # Clear progress indicators
            status_placeholder.empty()
            progress_placeholder.empty()
            
            return True
    except Exception as e:
        logger.error(f"Error in analysis workflow: {e}")
        st.error(f"An error occurred during analysis: {e}")
        return False

# Helper function to check DataFrame
def safe_check_dataframe(df):
    """Safely check if a DataFrame exists and has data"""
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

# -------------------------------------------------------------------------------
# Main function for the Try It Yourself tab
# -------------------------------------------------------------------------------
def render_try_yourself_tab(microsoft_colors):
    """Main tab rendering function"""
    # Initialize chat messages in session state if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Initialize other session state variables if needed
    for key, default in {
        'sentiment_results': None, 
        'business_lines': [],
        'stock_data': None,
        'date_col': None,
        'price_col': None,
        'ohlc_cols': None,
        'ticker_symbol': "MSFT", 
        'openai_api_key': "",
        'sentiment_timeline': None,
        'segment_data': {},
        'has_qa_section': False,
        'qa_results': None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # -----------------------------------------------------------------------
    # Header & Link Indicator
    # -----------------------------------------------------------------------
    st.markdown("🔗 ## Microsoft Earnings Call & Stock Analysis")
    
    # Create tabs for the analysis UI
    dashboard_tab, stock_tab, sentiment_tab, chatbot_tab = st.tabs([
        "Dashboard", "Stock Analysis", "Sentiment Analysis", "Chatbot Assistant"
    ])
    
    # -----------------------------------------------------------------------
    # Configuration Section - Similar to 3rd screenshot
    # -----------------------------------------------------------------------
    st.markdown("🔗 ## Analysis Configuration")
    
    # Show data source information - but don't allow changing it
    st.markdown(f"### Using Pre-categorized Data")
    st.code(f"Data source: {SENTIMENT_DATA_PATH}", language=None)
    
    # Layout the configuration options
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.text_input("Stock Ticker", value="MSFT", disabled=True)
    
    with col2:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
    
    with col3:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # OpenAI API key input - use password field
    st.text_input("OpenAI API Key", type="password", key="openai_api_key", 
                  help="Required for sentiment analysis. Your API key remains secured and is only used to process the sentences.")
    
    # Run analysis button
    if st.button("Run Analysis", type="primary", use_container_width=True):
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key")
        else:
            process_analysis_workflow(
                st.session_state.ticker_symbol,
                start_date,
                end_date
            )
    
    # Process chat input outside of all UI elements - at the bottom of the page
    # This avoids the Streamlit error about st.chat_input being used inside a container
    chat_prompt = None
    
    # Display and handle chat outside of tab containers to avoid errors
    if safe_check_dataframe(st.session_state.get('sentiment_results')):
        chat_prompt = st.chat_input("Ask about the earnings call or stock performance")
    
    # -----------------------------------------------------------------------
    # Dashboard Tab Content
    # -----------------------------------------------------------------------
    with dashboard_tab:
        if safe_check_dataframe(st.session_state.get('sentiment_results')) and safe_check_dataframe(st.session_state.get('stock_data')):
            st.subheader("Dashboard Overview")
            
            # Create a 2x2 dashboard layout
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                # Stock price summary card
                stock_data = st.session_state.stock_data
                price_col = st.session_state.price_col
                
                if safe_check_dataframe(stock_data) and price_col:
                    st.markdown("### Stock Performance")
                    
                    # Calculate key metrics
                    start_price = stock_data[price_col].iloc[0]
                    end_price = stock_data[price_col].iloc[-1]
                    percent_change = ((end_price - start_price) / start_price * 100)
                    
                    # Display metrics
                    metric_row1_1, metric_row1_2, metric_row1_3 = st.columns(3)
                    
                    with metric_row1_1:
                        st.metric("Starting Price", f"${start_price:.2f}")
                    
                    with metric_row1_2:
                        st.metric("Current Price", f"${end_price:.2f}", f"{percent_change:.2f}%")
                    
                    with metric_row1_3:
                        st.metric("Price Range", f"${stock_data[price_col].min():.2f} - ${stock_data[price_col].max():.2f}")
                    
                    # Simple stock chart
                    st.line_chart(stock_data.set_index(st.session_state.date_col)[price_col])
            
            with metric_col2:
                # Sentiment summary card
                sentiment_data = st.session_state.sentiment_results
                
                if safe_check_dataframe(sentiment_data):
                    st.markdown("### Sentiment Overview")
                    
                    # Calculate sentiment metrics
                    sentiment_counts = sentiment_data['sentiment'].value_counts()
                    total = len(sentiment_data)
                    
                    # Display metrics
                    metric_row2_1, metric_row2_2, metric_row2_3 = st.columns(3)
                    
                    with metric_row2_1:
                        pos_count = sentiment_counts.get('positive', 0)
                        pos_pct = (pos_count/total*100) if total > 0 else 0
                        st.metric("Positive", f"{pos_count} ({pos_pct:.1f}%)")
                    
                    with metric_row2_2:
                        neu_count = sentiment_counts.get('neutral', 0)
                        neu_pct = (neu_count/total*100) if total > 0 else 0
                        st.metric("Neutral", f"{neu_count} ({neu_pct:.1f}%)")
                    
                    with metric_row2_3:
                        neg_count = sentiment_counts.get('negative', 0)
                        neg_pct = (neg_count/total*100) if total > 0 else 0
                        st.metric("Negative", f"{neg_count} ({neg_pct:.1f}%)")
                    
                    # Sentiment chart
                    fig = plot_sentiment_distribution(sentiment_data, microsoft_colors)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="sent_dist_dash")
            
            # Combined sentiment and stock chart
            st.markdown("### Sentiment & Stock Correlation")
            
            if safe_check_dataframe(st.session_state.get('sentiment_timeline')):
                correlation_fig = plot_sentiment_and_stock(
                    st.session_state.sentiment_timeline, 
                    st.session_state.stock_data,
                    st.session_state.date_col,
                    st.session_state.price_col
                )
                if correlation_fig:
                    st.plotly_chart(correlation_fig, use_container_width=True, key="correlation_chart_dash")
                else:
                    st.warning("Could not create correlation chart")
            
            # Business segments breakdown
            st.markdown("### Business Segment Analysis")
            
            if 'business_line' in st.session_state.sentiment_results.columns:
                segment_fig = plot_sentiment_by_business_line(st.session_state.sentiment_results, microsoft_colors)
                if segment_fig:
                    st.plotly_chart(segment_fig, use_container_width=True, key="segment_chart_dash")
        else:
            st.info("No analysis results yet. Please run the analysis to see the dashboard.")
    
    # -----------------------------------------------------------------------
    # Stock Analysis Tab Content
    # -----------------------------------------------------------------------
    with stock_tab:
        if safe_check_dataframe(st.session_state.get('stock_data')):
            stock_data = st.session_state.stock_data
            date_col = st.session_state.date_col
            price_col = st.session_state.price_col
            ohlc_cols = st.session_state.ohlc_cols
            
            st.header("Stock Price Analysis")
            
            # Get min and max dates for the range selector
            min_date = stock_data[date_col].min().date()
            max_date = stock_data[date_col].max().date()
            
            # Date range selection
            st.subheader("Chart Settings")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                default_start = max(min_date, max_date - timedelta(days=365))
                start_date = st.date_input(
                    "Start Date (Stock Chart)",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="stock_start_date"
                )
            
            with col2:
                end_date = st.date_input(
                    "End Date (Stock Chart)",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="stock_end_date"
                )
            
            # Chart type selection
            with col3:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Line", "Candlestick", "OHLC"],
                    index=0
                )
            
            # Show moving averages and earnings dates options
            col4, col5 = st.columns(2)
            
            with col4:
                show_ma = st.checkbox("Show Moving Averages", value=True)
            
            with col5:
                show_earnings = st.checkbox("Show Earnings Dates", value=True)
            
            # Make sure end_date is not before start_date
            if start_date > end_date:
                st.error("End date must be after start date")
                end_date = start_date
            
            # Filter data based on selected date range
            mask = (stock_data[date_col].dt.date >= start_date) & (stock_data[date_col].dt.date <= end_date)
            filtered_df = stock_data[mask].copy()
            
            # Show how many data points are in the filtered range
            st.write(f"Showing {len(filtered_df)} data points from {start_date} to {end_date}")
            
            # Create the main chart based on selected type
            if chart_type == "Line":
                # Simple line chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_df[date_col],
                    y=filtered_df[price_col],
                    mode='lines',
                    name='MSFT',
                    line=dict(color='#0078D4', width=2)  # Microsoft blue
                ))
                
                # Add moving averages if requested
                if show_ma and len(filtered_df) > 50:
                    # 20-day moving average
                    filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
                    fig.add_trace(go.Scatter(
                        x=filtered_df[date_col],
                        y=filtered_df['MA20'],
                        mode='lines',
                        name='20-Day MA',
                        line=dict(color='#107C10', width=1.5, dash='dash')  # Microsoft green
                    ))
                    
                    # 50-day moving average if enough data
                    if len(filtered_df) > 100:
                        filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                        fig.add_trace(go.Scatter(
                            x=filtered_df[date_col],
                            y=filtered_df['MA50'],
                            mode='lines',
                            name='50-Day MA',
                            line=dict(color='#5C2D91', width=1.5, dash='dash')  # Microsoft purple
                        ))
                
            elif chart_type == "Candlestick":
                # Need OHLC data for candlestick
                if ohlc_cols and 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=filtered_df[date_col],
                        open=filtered_df[ohlc_cols['open']],
                        high=filtered_df[ohlc_cols['high']],
                        low=filtered_df[ohlc_cols['low']],
                        close=filtered_df[price_col],
                        name='MSFT',
                        increasing=dict(line=dict(color='#107C10')),  # Microsoft green
                        decreasing=dict(line=dict(color='#D13438'))   # Microsoft red
                    )])
                    
                    # Add moving averages if requested
                    if show_ma and len(filtered_df) > 50:
                        # 20-day moving average
                        filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(
                            x=filtered_df[date_col],
                            y=filtered_df['MA20'],
                            mode='lines',
                            name='20-Day MA',
                            line=dict(color='#FFFFFF', width=1.5, dash='dash')  # White for contrast
                        ))
                        
                        # 50-day moving average if enough data
                        if len(filtered_df) > 100:
                            filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                            fig.add_trace(go.Scatter(
                                x=filtered_df[date_col],
                                y=filtered_df['MA50'],
                                mode='lines',
                                name='50-Day MA',
                                line=dict(color='#FFB900', width=1.5, dash='dash')  # Microsoft yellow
                            ))
                else:
                    st.warning("OHLC data not found, defaulting to line chart")
                    fig = px.line(filtered_df, x=date_col, y=price_col, title=f"MSFT Stock Price ({start_date} to {end_date})")
            
            elif chart_type == "OHLC":
                # Similar to candlestick but different visual style
                if ohlc_cols and 'open' in ohlc_cols and 'high' in ohlc_cols and 'low' in ohlc_cols:
                    # Create OHLC chart
                    fig = go.Figure(data=[go.Ohlc(
                        x=filtered_df[date_col],
                        open=filtered_df[ohlc_cols['open']],
                        high=filtered_df[ohlc_cols['high']],
                        low=filtered_df[ohlc_cols['low']],
                        close=filtered_df[price_col],
                        name='MSFT',
                        increasing=dict(line=dict(color='#107C10')),  # Microsoft green
                        decreasing=dict(line=dict(color='#D13438'))   # Microsoft red
                    )])
                    
                    # Add moving averages if requested
                    if show_ma and len(filtered_df) > 50:
                        # 20-day moving average
                        filtered_df['MA20'] = filtered_df[price_col].rolling(window=20).mean()
                        fig.add_trace(go.Scatter(
                            x=filtered_df[date_col],
                            y=filtered_df['MA20'],
                            mode='lines',
                            name='20-Day MA',
                            line=dict(color='#FFFFFF', width=1.5, dash='dash')  # White for contrast
                        ))
                        
                        # 50-day moving average if enough data
                        if len(filtered_df) > 100:
                            filtered_df['MA50'] = filtered_df[price_col].rolling(window=50).mean()
                            fig.add_trace(go.Scatter(
                                x=filtered_df[date_col],
                                y=filtered_df['MA50'],
                                mode='lines',
                                name='50-Day MA',
                                line=dict(color='#FFB900', width=1.5, dash='dash')  # Microsoft yellow
                            ))
                else:
                    st.warning("OHLC data not found, defaulting to line chart")
                    fig = px.line(filtered_df, x=date_col, y=price_col, title=f"MSFT Stock Price ({start_date} to {end_date})")
            
            # Update layout
            fig.update_layout(
                title=f"MSFT Stock Price ({start_date} to {end_date})",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=600,
                template="plotly_white",
                xaxis_rangeslider_visible=False,  # Hide range slider for cleaner look
                hovermode="x unified"
            )
            
            # Add earnings date vertical lines if requested
            if show_earnings:
                # Convert hardcoded dates to datetime for comparison
                earnings_dates = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in MSFT_EARNINGS_DATES]
                
                # Filter to only show earnings dates in the selected range
                range_earnings_dates = [date for date in earnings_dates 
                                      if start_date <= date <= end_date]
                
                if range_earnings_dates:
                    # Find min and max prices for proper line positioning
                    y_min = filtered_df[price_col].min()
                    y_max = filtered_df[price_col].max()
                    y_range = y_max - y_min
                    
                    for earnings_date in range_earnings_dates:
                        # Find closest trading day to earnings date
                        closest_date = find_closest_trading_day(filtered_df, date_col, earnings_date)
                        
                        if not closest_date:
                            continue
                        
                        # Add vertical line for the earnings date
                        fig.add_vline(
                            x=closest_date,
                            line=dict(
                                color="#FFB900",
                                width=1.5,
                                dash="dash"
                            )
                        )
                        
                        # Add annotation for the earnings date
                        fig.add_annotation(
                            x=closest_date,
                            y=y_max + (y_range * 0.02),  # Position above the highest point
                            text=f"Earnings: {closest_date.strftime('%Y-%m-%d')}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1.5,
                            arrowcolor="#FFB900",
                            font=dict(color="#FFB900", size=10),
                            align="center"
                        )
            
            # Display the main chart
            st.plotly_chart(fig, use_container_width=True, key="stock_price_chart")
            
            # Earnings impact analysis section
            st.subheader("Earnings Call Impact Analysis")
            
            # Earnings date selector
            earnings_dates = [datetime.strptime(date_str, "%Y-%m-%d").date() for date_str in MSFT_EARNINGS_DATES]
            range_earnings_dates = [date for date in earnings_dates if start_date <= date <= end_date]
            
            if range_earnings_dates:
                date_options = [date.strftime("%Y-%m-%d") for date in range_earnings_dates]
                
                selected_earnings = st.selectbox(
                    "Select Earnings Date to Analyze", 
                    date_options
                )
                
                # Analyze selected earnings date
                if selected_earnings:
                    highlight_date = datetime.strptime(selected_earnings, "%Y-%m-%d").date()
                    create_earnings_impact_analysis(filtered_df, date_col, price_col, ohlc_cols, highlight_date)
            else:
                st.info("No earnings dates found in the selected date range")
            
            # Summary statistics
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Layout statistics in columns
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            # Calculate statistics
            start_price = filtered_df[price_col].iloc[0] if not filtered_df.empty else 0
            end_price = filtered_df[price_col].iloc[-1] if not filtered_df.empty else 0
            percent_change = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0
            min_price = filtered_df[price_col].min() if not filtered_df.empty else 0
            max_price = filtered_df[price_col].max() if not filtered_df.empty else 0
            
            # Display statistics with custom HTML metrics
            with stat_col1:
                st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Start Price</div>"
                           f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${start_price:.2f}</div>", 
                           unsafe_allow_html=True)
            
            # Use custom colored_metric for End Price
            with stat_col2:
                colored_metric(
                    "End Price", 
                    f"${end_price:.2f}", 
                    percent_change
                )
            
            with stat_col3:
                st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Min Price</div>"
                           f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${min_price:.2f}</div>", 
                           unsafe_allow_html=True)
            
            with stat_col4:
                st.markdown(f"<div style='font-size: 0.8rem; color: rgba(49, 51, 63, 0.6);'>Max Price</div>"
                           f"<div style='font-size: 1.5rem; font-weight: 700; margin: 0.2rem 0;'>${max_price:.2f}</div>", 
                           unsafe_allow_html=True)
            
            # Option to show the data table
            if st.checkbox("Show Data Table", value=False):
                st.dataframe(filtered_df)
        else:
            st.info("No stock data available. Please run the analysis first.")
    
    # -----------------------------------------------------------------------
    # Sentiment Analysis Tab Content
    # -----------------------------------------------------------------------
    with sentiment_tab:
        if safe_check_dataframe(st.session_state.get('sentiment_results')):
            sentiment_data = st.session_state.sentiment_results
            
            st.header("Earnings Call Sentiment Analysis")
            
            # Overall sentiment section
            st.subheader("Overall Sentiment Distribution")
            
            # Display summary metrics
            sentiment_counts = sentiment_data['sentiment'].value_counts()
            total = len(sentiment_data)
            
            sent_col1, sent_col2, sent_col3 = st.columns(3)
            with sent_col1:
                pos_count = sentiment_counts.get('positive', 0)
                pos_pct = (pos_count/total*100) if total > 0 else 0
                st.metric("Positive", f"{pos_count} ({pos_pct:.1f}%)")
            
            with sent_col2:
                neu_count = sentiment_counts.get('neutral', 0)
                neu_pct = (neu_count/total*100) if total > 0 else 0
                st.metric("Neutral", f"{neu_count} ({neu_pct:.1f}%)")
            
            with sent_col3:
                neg_count = sentiment_counts.get('negative', 0)
                neg_pct = (neg_count/total*100) if total > 0 else 0
                st.metric("Negative", f"{neg_count} ({neg_pct:.1f}%)")
            
            # Create sentiment distribution chart
            fig = plot_sentiment_distribution(sentiment_data, microsoft_colors)
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="sent_dist_overall")
            
            # If we have a Q&A section, show that analysis separately
            if st.session_state.has_qa_section and safe_check_dataframe(st.session_state.get('qa_results')):
                st.markdown("### Q&A Section Sentiment")
                
                # Q&A summary metrics
                qa_sentiment_counts = st.session_state.qa_results['sentiment'].value_counts()
                qa_total = len(st.session_state.qa_results)
                
                qa_col1, qa_col2, qa_col3 = st.columns(3)
                with qa_col1:
                    pos_count = qa_sentiment_counts.get('positive', 0)
                    pos_pct = (pos_count/qa_total*100) if qa_total > 0 else 0
                    st.metric("Positive", f"{pos_count} ({pos_pct:.1f}%)")
                
                with qa_col2:
                    neu_count = qa_sentiment_counts.get('neutral', 0)
                    neu_pct = (neu_count/qa_total*100) if qa_total > 0 else 0
                    st.metric("Neutral", f"{neu_count} ({neu_pct:.1f}%)")
                
                with qa_col3:
                    neg_count = qa_sentiment_counts.get('negative', 0)
                    neg_pct = (neg_count/qa_total*100) if qa_total > 0 else 0
                    st.metric("Negative", f"{neg_count} ({neg_pct:.1f}%)")
                
                # Q&A sentiment chart
                qa_fig = plot_sentiment_distribution(st.session_state.qa_results, microsoft_colors, "Q&A Section Sentiment")
                if qa_fig:
                    st.plotly_chart(qa_fig, use_container_width=True, key="sent_dist_qa")
            
            # Business segments section
            st.markdown("---")
            st.subheader("Sentiment by Business Segment")
            
            # Create business segment chart
            segment_fig = plot_sentiment_by_business_line(sentiment_data, microsoft_colors)
            if segment_fig:
                st.plotly_chart(segment_fig, use_container_width=True, key="segment_chart_sentiment")
            
            # Detailed sentiment analysis section
            st.markdown("---")
            st.subheader("Detailed Sentiment Analysis")
            
            # Show business line selector
            if 'business_line' in sentiment_data.columns:
                # Get business lines with counts for better display
                bl_counts = sentiment_data['business_line'].value_counts()
                business_lines = []
                
                for bl in sorted(sentiment_data['business_line'].unique()):
                    count = bl_counts.get(bl, 0)
                    business_lines.append(f"{bl} ({count})")
                
                selected_bl_with_count = st.selectbox(
                    "Select Business Line",
                    ["All Business Lines"] + business_lines
                )
                
                # Extract actual business line name without count
                if selected_bl_with_count == "All Business Lines":
                    selected_bl = "All Business Lines"
                else:
                    selected_bl = selected_bl_with_count.split(" (")[0]
                
                # Filter by selected business line
                if selected_bl == "All Business Lines":
                    filtered_results = sentiment_data.copy()
                else:
                    filtered_results = sentiment_data[
                        sentiment_data['business_line'] == selected_bl
                    ].copy()
                
                # Enhanced search options
                search_col1, search_col2 = st.columns([3, 1])
                
                with search_col1:
                    search_term = st.text_input("Search in statements:")
                
                with search_col2:
                    sentiment_filter = st.selectbox(
                        "Filter by sentiment",
                        ["All", "Positive", "Neutral", "Negative"]
                    )
                
                # Apply search filter if provided
                if search_term:
                    display_results = filtered_results[
                        filtered_results['text'].str.contains(search_term, case=False, na=False)
                    ].copy()
                else:
                    display_results = filtered_results.copy()
                
                # Apply sentiment filter if selected
                if sentiment_filter != "All":
                    sentiment_value = sentiment_filter.lower()
                    display_results = display_results[display_results['sentiment'] == sentiment_value].copy()
                
                # Show filtered results count with better styling
                st.markdown(
                    f"""
                    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 10px;">
                    Showing <b>{len(display_results)}</b> of <b>{len(filtered_results)}</b> statements from 
                    <b>{selected_bl}</b> with sentiment <b>{sentiment_filter}</b>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Display sentiment summary for selection with icons
                sentiment_counts = display_results['sentiment'].value_counts()
                total = len(display_results)
                
                if total > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pos_count = sentiment_counts.get('positive', 0)
                        pos_pct = (pos_count/total*100) if total > 0 else 0
                        st.metric("✅ Positive", f"{pos_count} ({pos_pct:.1f}%)")
                    
                    with col2:
                        neu_count = sentiment_counts.get('neutral', 0)
                        neu_pct = (neu_count/total*100) if total > 0 else 0
                        st.metric("⚖️ Neutral", f"{neu_count} ({neu_pct:.1f}%)")
                    
                    with col3:
                        neg_count = sentiment_counts.get('negative', 0)
                        neg_pct = (neg_count/total*100) if total > 0 else 0
                        st.metric("⚠️ Negative", f"{neg_count} ({neg_pct:.1f}%)")
                
                # Display statements with improved pagination
                st.markdown("#### Analyzed Statements")
                
                if display_results.empty:
                    st.info("No statements match your search criteria")
                else:
                    # Display statements with pagination
                    page_size = 10
                    total_pages = (len(display_results) + page_size - 1) // page_size
                    
                    if total_pages > 0:
                        # Create columns for pagination controls
                        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                        
                        with page_col2:
                            if total_pages > 1:
                                # normal slider when there is more than one page
                                page_num = st.slider(
                                    "Page",
                                    1,
                                    total_pages,
                                    1,
                                    key="detail_page_slider"
                                )
                            else:
                                # only one page → skip the slider
                                page_num = 1
                                st.markdown("Page 1&nbsp;of&nbsp;1")  # small note so the layout doesn’t shift
                        
                        # Calculate which sentences to show
                        start_idx = (page_num - 1) * page_size
                        end_idx = min(start_idx + page_size, len(display_results))
                        
                        # Show current page info
                        st.markdown(f"Showing statements {start_idx+1}-{end_idx} of {len(display_results)}")
                        
                        # Display statements with improved styling
                        for idx in range(start_idx, end_idx):
                            if idx < len(display_results):
                                row = display_results.iloc[idx]
                                
                                # Get appropriate sentiment color and icon
                                sentiment_color = {
                                    'positive': microsoft_colors['success'],
                                    'neutral': microsoft_colors['accent'],
                                    'negative': microsoft_colors['danger']
                                }.get(row['sentiment'], microsoft_colors['accent'])
                                
                                sentiment_icon = {
                                    'positive': "✅",
                                    'neutral': "⚖️",
                                    'negative': "⚠️"
                                }.get(row['sentiment'], "")
                                
                                # Format the statement with improved styling
                                st.markdown(
                                    f"""
                                    <div style='padding: 15px; margin: 10px 0; border-left: 4px solid {sentiment_color}; 
                                    background-color: rgba({int(sentiment_color[1:3], 16)}, {int(sentiment_color[3:5], 16)}, {int(sentiment_color[5:7], 16)}, 0.05);
                                    border-radius: 0 5px 5px 0;'>
                                    <div style='display: flex; justify-content: space-between;'>
                                        <span style='font-weight: bold;'>{row['business_line'] if 'business_line' in row else ''}</span>
                                        <span style='color: {sentiment_color};'>{sentiment_icon} {row['sentiment'].upper() if 'sentiment' in row and row['sentiment'] else 'N/A'}</span>
                                    </div>
                                    <div style='margin-top: 8px;'>{row['text'] if 'text' in row else ''}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No statements match your search criteria")
            else:
                st.warning("No business line information available in the sentiment data")
        else:
            st.info("No sentiment data available. Please run the analysis first.")
    
    # -----------------------------------------------------------------------
    # Chatbot Tab Content
    # -----------------------------------------------------------------------
    with chatbot_tab:
        st.header("Financial Analysis Chatbot")
        
        # Check if necessary data is available
        has_sentiment = safe_check_dataframe(st.session_state.get('sentiment_results'))
        has_stock = safe_check_dataframe(st.session_state.get('stock_data'))
        
        if has_sentiment or has_stock:
            st.markdown("""
            Ask questions about the earnings call sentiment analysis and stock data. Examples:
            - What is the overall sentiment in the earnings call?
            - How did the stock perform after the last earnings?
            - Which business segment had the most positive sentiment?
            - What was the stock price movement around earnings dates?
            """)
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.info("Please run the analysis first to enable the chatbot.")
            st.markdown("The chatbot needs sentiment analysis and stock data to answer your questions effectively.")
    
    # Handle chat outside of tabs to avoid container errors
    if chat_prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": chat_prompt})
        
        # Check if OpenAI API key is available
        if not st.session_state.openai_api_key:
            # Add assistant response about missing API key
            st.session_state.messages.append({"role": "assistant", "content": "Please enter your OpenAI API key in the configuration section to use the chatbot."})
        else:
            # Get message response from OpenAI
            with st.spinner("Thinking..."):
                response = analyze_chat_query(
                    chat_prompt, 
                    st.session_state.sentiment_results if has_sentiment else None,
                    st.session_state.stock_data if has_stock else None,
                    st.session_state.date_col if has_stock else None,
                    st.session_state.price_col if has_stock else None
                )
                
                if response:
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    error_msg = "I'm having trouble answering that. Please try again or rephrase your question."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to display the updated chat
        st.rerun()

# If the file is run directly, set up a test environment
if __name__ == "__main__":
    st.set_page_config(
        page_title="Microsoft Earnings Call Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Define Microsoft colors for the standalone batch
    microsoft_colors = {
        'primary': '#0078d4',    # Microsoft blue
        'secondary': '#5c2d91',  # Microsoft purple
        'success': '#107c10',    # Microsoft green
        'danger': '#d13438',     # Microsoft red
        'warning': '#ffb900',    # Microsoft yellow
        'info': '#00b7c3',       # Microsoft teal
        'accent': '#5c2d91',     # Microsoft purple
    }
    
    # Call the main function
    render_try_yourself_tab(microsoft_colors)
