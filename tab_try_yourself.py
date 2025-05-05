import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Sample data generation for demo
def generate_sample_data(days=30):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    # Stock prices with an overall upward trend but some fluctuations
    stock_prices = np.linspace(100, 130, days) + np.random.normal(0, 5, days)
    
    # Sentiment scores between -1 and 1 with correlation to stock price
    sentiment_base = np.linspace(-0.5, 0.8, days)
    sentiment_noise = np.random.normal(0, 0.2, days)
    sentiment = np.clip(sentiment_base + sentiment_noise, -1, 1)
    
    return pd.DataFrame({
        'Date': dates,
        'StockPrice': stock_prices,
        'Sentiment': sentiment
    })

def render_try_yourself_tab(microsoft_colors):
    st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
    st.markdown("## Analyze Earnings Call Transcripts")
    st.markdown("Upload your earnings call transcript to generate sentiment analysis and explore correlations with stock performance.")
    
    # Create a placeholder for file upload
    uploaded_file = st.file_uploader("Upload Earnings Call Transcript", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Placeholder for processing
        with st.spinner("Analyzing transcript..."):
            st.info("This would typically process the uploaded file, but we're showing demo data for now.")
            
            # In a real app, this is where you'd process the file
            # For now, let's just display a sample
            df = generate_sample_data()
            
            # Display sample content of the file
            st.markdown("### Preview of Uploaded Transcript")
            st.text("SAMPLE EARNINGS CALL TRANSCRIPT\n\nOperator: Good morning, and welcome to the Q1 2025 Earnings Conference Call...")
    
    else:
        st.info("Please upload an earnings call transcript to begin analysis.")
        st.markdown("You can try a sample analysis by clicking the button below:")
        if st.button("Run Sample Analysis"):
            st.success("Running sample analysis!")
            df = generate_sample_data()
        else:
            df = None
            
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Results section
    if 'df' in locals() and df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("### Stock Price vs Time")
            fig = px.line(df, x='Date', y='StockPrice', markers=True,
                          title='Stock Price Trend',
                          labels={'StockPrice': 'Stock Price ($)', 'Date': 'Date'},
                          template='plotly_white')
            fig.update_traces(line_color=microsoft_colors["primary"], line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            st.markdown("### Sentiment Analysis Over Time")
            fig = px.line(df, x='Date', y='Sentiment', markers=True,
                          title='Sentiment Trend',
                          labels={'Sentiment': 'Sentiment Score', 'Date': 'Date'},
                          template='plotly_white')
            
            # Color based on sentiment (red for negative, green for positive)
            fig.update_traces(line_color=microsoft_colors["primary"], line_width=2)
            
            # Add a horizontal line at y=0
            fig.add_shape(
                type='line',
                y0=0, y1=0,
                x0=df['Date'].min(), x1=df['Date'].max(),
                line=dict(color='gray', dash='dash')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional statistics
        st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
        st.markdown("### Key Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sentiment = df['Sentiment'].mean()
            sentiment_color = microsoft_colors["success"] if avg_sentiment > 0 else microsoft_colors["danger"]
            st.markdown(f"""
            <div style="text-align: center;">
                <h4>Average Sentiment</h4>
                <p style="font-size: 24px; color: {sentiment_color};">{avg_sentiment:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            sentiment_volatility = df['Sentiment'].std()
            st.markdown(f"""
            <div style="text-align: center;">
                <h4>Sentiment Volatility</h4>
                <p style="font-size: 24px; color: {microsoft_colors["primary"]};">{sentiment_volatility:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            stock_change = (df['StockPrice'].iloc[-1] - df['StockPrice'].iloc[0]) / df['StockPrice'].iloc[0] * 100
            stock_color = microsoft_colors["success"] if stock_change > 0 else microsoft_colors["danger"]
            st.markdown(f"""
            <div style="text-align: center;">
                <h4>Stock Change</h4>
                <p style="font-size: 24px; color: {stock_color};">{stock_change:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            correlation = df['Sentiment'].corr(df['StockPrice'])
            corr_color = microsoft_colors["primary"]
            st.markdown(f"""
            <div style="text-align: center;">
                <h4>Correlation</h4>
                <p style="font-size: 24px; color: {corr_color};">{correlation:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
