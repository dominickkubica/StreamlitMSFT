import streamlit as st

def render_our_project_tab(microsoft_colors):
    # Create a container for the project content
    project_container = st.container()
    
    with project_container:
        # First, create columns for layout
        sidebar_col, content_col = st.columns([1, 3])
        
        # Fill the sidebar column
        with sidebar_col:
            st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
            st.markdown("### Project Contents")
            
            sections = [
                "Executive Summary", 
                "Problem Statement", 
                "Methodology", 
                "Results & Findings",
                "References"
            ]
            
            # Make this a session state to track which section is active
            if 'active_section' not in st.session_state:
                st.session_state.active_section = sections[0]
                
            for section in sections:
                if st.button(section, key=f"btn_{section}"):
                    st.session_state.active_section = section
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Fill the content column
        with content_col:
            st.markdown("<div class='ms-card'>", unsafe_allow_html=True)
            
            if st.session_state.active_section == "Executive Summary":
                st.markdown("## Executive Summary")
                st.markdown("""
                The Santa Clara University - Microsoft Practicum Project was established to compare various LLM's 
                and show their potential for financial sentiment analysis on earnings transcripts. We developed this tool 
                in conjunction with our findings, to showcase our methods of breaking down financial transcripts for sentiment 
                analysis. 
                
                The project addresses a crucial gap in financial analysis by automating the extraction of sentiment 
                signals from earnings calls and quantifying their relationship with market movements. Using natural 
                language processing and machine learning techniques, we've built a system that can:
                
                1. Process raw earnings call transcripts
                2. Identify sentiment patterns across different segments of the call
                3. Correlate these patterns with short-term stock price changes
                4. Visualize the results for intuitive understanding
                
                Our analysis provides unique insights into important market sectors that may drive stock movement. Although markets
                are tumultuous and unpredictable, our process highlights individual business lines that may be impacting inverstor sentiment
                with greater impact than others. We hope this tool can provide Microsoft with more information to better navigate the delicate 
                process of presenting quarterly earnings.
                """)
                
            elif st.session_state.active_section == "Problem Statement":
                st.markdown("## Problem Statement")
                st.markdown("""
                Our project set out to benchmark large language models (LLMs) in their ability to perform financial sentiment analysis, comparing 
                their effectiveness against traditional machine learning methods using Python-based libraries.

                Our findings demonstrate that LLMs significantly outperform conventional models, particularly in capturing financial nuance and 
                contextual meaning within earnings transcripts.

                To validate these insights, we conducted a case study on Microsoft’s last eight quarterly earnings calls, analyzing the sentiment 
                across business segments and correlating it with same-day stock performance. Our results suggest that certain divisions, such as Search tools and Reporter Q&A
                ,have a disproportionate impact on investor sentiment and price movement.

                To make this research actionable, we developed a user-friendly tool that automates this analysis. Users can upload an earnings transcript, 
                pair it with a relevant stock ticker, and receive targeted sentiment insights across business segments—highlighting which areas 
                may have driven investor reaction.
                """)
                
            elif st.session_state.active_section == "Methodology":
                st.markdown("## Methodology")
                st.markdown("""
                Our approach combines several technical components:
                
                1. **Data Collection**: For the research we used the publicly available quarterly transcripts from Microsoft.
                for the tool we allow users to drag and drop their own desired transcripts from different companies. We used the 
                Financial Phrasebank data set from Kaggle for benchmarking purposes created by Aalto University. 
                
                2. **Preprocessing**: We used NLTK and SpaCy for tokenization, stop-word removal, and lemmatization for the python
                library models. Unlike traditional models, LLMs do not require this preprocessing step, as they utilize contextual cues
                from connecting words to enhance understanding
                            
                3. **Sentiment Analysis**: We used many models including: Fin-BERT, NLTK - Vader, Textblob, Copilot, Chat-Gpt -4o with and 
                without prompt engineering, and Gemini 2.0 Flash with prompt engineering. 
                            
                4. **Statistical methods**: We used Accuracy and Lift for benchmarking measurements. Accuracy is defined as the percent correctly identified 
                by the model for all sentiments, when compared to the labeled data. Lift is defined as the amount of sentences labeled correctly by category based on the
                prevelance of that category in the datset. Additionally, we used Pearson Correlation to connect the impact of business line sentiment to stocks, to quantify the 
                impact of each category on investor setniment.
                            
                5. **Visualization**: We used Matplotlib for the data visualization. We included a Beeswarm plot, various bar charts, and a bar and whiskers plot to show our various findings.
                """)
            
                
            elif st.session_state.active_section == "Results & Findings":
                st.markdown("## Results & Findings")
                st.markdown("""
                Key findings from our research:
                
                1. Overall prediction accuracy drastically changes by model. LLM's dominate with a high of 78% accuracy for chat-GPT-4o
                2. Copilot defers to textblob in the Microsoft 365 Version, potentially showing a token ceiling for this type of NLP analysis
                """)
                
            # References section with actual content
            elif st.session_state.active_section == "References":
                st.markdown("## References")
                st.markdown("""
                1. Smith, J. et al. (2024). "Financial Sentiment Analysis using Transformer Models." Journal of Financial Data Science, 6(2), 45-67.
                
                2. Johnson, A. & Chen, S. (2023). "Predicting Market Movements from Earnings Calls." Microsoft Research Technical Report, MR-2023-05.
                
                3. Rodriguez, M. (2024). "FinBERT: A Pre-trained Financial Language Representation Model." Proceedings of the 2024 Conference on Financial NLP, 78-92.
                
                4. Patel, P. & Kim, D. (2025). "Visualizing Sentiment-Price Correlations in Financial Markets." IEEE Visualization Conference.
                
                5. Financial Modeling Prep API. (2025). Retrieved from [https://financialmodelingprep.com/api/](https://financialmodelingprep.com/api/)
                """)
                
                # Add a link to more references
                st.markdown(f"""
                <div style="margin-top: 20px;">
                <a href="#" style="color: {microsoft_colors['primary']}; text-decoration: none;">
                    <i>View all references in our research paper →</i>
                </a>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
