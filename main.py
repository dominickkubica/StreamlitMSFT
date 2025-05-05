import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Import tab modules
from tab_try_yourself import render_try_yourself_tab
from tab_about_us import render_about_us_tab
from tab_our_project import render_our_project_tab

# Set page configuration
st.set_page_config(
    page_title="Reading Between the Lines with AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div.block-container {padding-top: 1rem;}
    div.stTooltip {display: none;}
    button[data-baseweb="tooltip"] div {display: none;}
    button[title] {position: relative;}
    button[title]:hover::after {display: none;}
    </style>
    """, unsafe_allow_html=True)

# Define Microsoft theme colors (shared across all tabs)
microsoft_colors = {
    "primary": "#0078d4",  # Microsoft Blue
    "secondary": "#50e6ff",  # Light Blue
    "accent": "#ffb900",  # Yellow
    "success": "#107c10",  # Green
    "danger": "#d13438",  # Red
    "background": "#f3f2f1",  # Light Gray
    "text": "#323130"  # Dark Gray
}

# Apply Microsoft styling
def apply_ms_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {microsoft_colors["background"]};
            color: {microsoft_colors["text"]};
        }}
        .stButton>button {{
            background-color: {microsoft_colors["primary"]};
            color: white;
            border-radius: 2px;
        }}
        .stButton>button:hover {{
            background-color: {microsoft_colors["secondary"]};
            color: {microsoft_colors["text"]};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: white;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            border: none;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {microsoft_colors["primary"]};
            color: white;
        }}
        .css-145kmo2 {{
            font-family: 'Segoe UI', sans-serif;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Segoe UI', sans-serif;
            color: {microsoft_colors["primary"]};
        }}
        .ms-card {{
            background-color: white;
            border-radius: 4px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }}
        .profile-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background-color: white;
            margin: 10px;
            height: 100%;
        }}
        .profile-img {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            object-fit: cover;
            margin-bottom: 15px;
            border: 3px solid {microsoft_colors["primary"]};
        }}
        .contact-form {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .sidebar-nav {{
            background-color: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}
        .sidebar-item {{
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 4px;
            cursor: pointer;
        }}
        .sidebar-item:hover {{
            background-color: {microsoft_colors["secondary"]};
            color: {microsoft_colors["text"]};
        }}
        .sidebar-item.active {{
            background-color: {microsoft_colors["primary"]};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Microsoft Logo
def display_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <svg width="40" height="40" viewBox="0 0 23 23">
                    <rect x="1" y="1" width="10" height="10" fill="{microsoft_colors["primary"]}"/>
                    <rect x="12" y="1" width="10" height="10" fill="{microsoft_colors["success"]}"/>
                    <rect x="1" y="12" width="10" height="10" fill="{microsoft_colors["accent"]}"/>
                    <rect x="12" y="12" width="10" height="10" fill="{microsoft_colors["danger"]}"/>
                </svg>
                <h1 style="margin-left: 10px; color: {microsoft_colors["primary"]};">Reading Between the Lines</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

# Define photo paths directly (shared across tabs)
def get_team_photo_paths():
    # Using built-in Streamlit image display which handles URLs better
    photo_paths = {
        "Dylan Gordon": "https://i.imgur.com/cZdnGVu.png", 
        "Dominick Kubica": "https://i.imgur.com/D3uLwZy.png",
        "Nanami Emura": "https://i.imgur.com/1tlJMJO.png",
        "Derleen Saini": "https://i.imgur.com/JA4pDjr.png",
        "Charles Goldenberg": "https://via.placeholder.com/150"
    }
    return photo_paths

# Function to safely load profile images (shared across tabs)
def display_profile_image(name):
    try:
        img_url = get_team_photo_paths().get(name)
        if img_url:
            st.image(img_url, width=150)
        else:
            st.image("https://via.placeholder.com/150", width=150)
    except Exception:
        st.image("https://via.placeholder.com/150", width=150)

# Main application
def main():
    # Apply Microsoft theme
    apply_ms_theme()
    
    # Display header
    display_header()
    
    # Create navigation tabs
    tabs = st.tabs(["Try It Yourself", "About Us", "Our Project"])
    
    # Tab 1: Try It Yourself
    with tabs[0]:
        render_try_yourself_tab(microsoft_colors)
    
    # Tab 2: About Us
    with tabs[1]:
        render_about_us_tab(microsoft_colors, display_profile_image)
    
    # Tab 3: Our Project
    with tabs[2]:
        render_our_project_tab(microsoft_colors)

if __name__ == "__main__":
    main()
