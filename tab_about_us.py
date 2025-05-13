import streamlit as st
from pathlib import Path
import base64
from PIL import Image, ImageDraw

###############################################################################
#  Photo location helpers
###############################################################################
BASE_DIR = Path(__file__).resolve().parent
# Create a specific assets directory for images
ASSETS_DIR = BASE_DIR / "assets"

# Ensure the assets directory exists
ASSETS_DIR.mkdir(exist_ok=True)

PHOTO_MAP = {
    "Dylan Gordon":       "dylan.png",
    "Dominick Kubica":    "dominick.png",
    "Nanami Emura":       "nanmi.jpg",
    "Derleen Saini":      "derleen.jpg",
    "Charles Goldenberg": "charles.png",
}

def get_image_as_base64(file_path):
    """Convert an image file to base64 for reliable display"""
    try:
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return encoded
    except Exception as e:
        # Use print instead of st.debug
        print(f"Error encoding image {file_path}: {e}")
        return None

def display_profile_image(name: str, width: int | None = None):
    """
    Display a teammate's photo or a gray placeholder if the file is missing.
    """
    # 1) Exact mapping first
    filename = PHOTO_MAP.get(name)

    # 2) If not in map: try FIRSTNAME.png (legacy convenience)
    if filename is None:
        filename = f"{name.split()[0].upper()}.png"

    # Try multiple locations for the image file
    possible_paths = [
        ASSETS_DIR / filename,  # Check assets directory first
        BASE_DIR / filename,    # Then check main directory
        Path(filename)          # Finally check relative path
    ]
    
    img_found = False
    for img_path in possible_paths:
        if img_path.exists():
            # Use base64 encoding for reliable image display
            encoded_image = get_image_as_base64(img_path)
            if encoded_image:
                html = f'<img src="data:image/png;base64,{encoded_image}" ' \
                       f'width="{width if width else "100%"}" ' \
                       f'style="max-width:100%; border-radius:5px;">'
                st.markdown(html, unsafe_allow_html=True)
                img_found = True
                break
    
    if not img_found:
        # Draw simple placeholder
        size = 150
        ph = Image.new("RGB", (size, size), color="#CCCCCC")
        d = ImageDraw.Draw(ph)
        d.text((size * 0.25, size * 0.4), "No\nPhoto", fill="black")
        st.image(ph, use_column_width=(width is None), width=width)
        
        # Use print instead of st.debug - won't show in UI but will appear in console
        print(f"No image found for {name}. Checked paths: {possible_paths}")
        
        # Alternatively, if you want to show debugging info in development:
        # if st.session_state.get('debug_mode', False):
        #     st.warning(f"No image found for {name}")

###############################################################################
#  Main component
###############################################################################
def render_about_us_tab(microsoft_colors: dict):
    st.markdown("## Meet Our Team")
    st.markdown(
        "We are a group of passionate data scientists and financial analysts "
        "working to revolutionize how earnings calls are analyzed."
    )

    team_members = [
        # (unchanged) ----------------------------------------------------------
        {
            "name": "Dylan Gordon",
            "role": "University Researcher",
            "about": "Dylan is a former chemical engineer turned data scientist. "
                     "He specializes in optimization and machine learning.",
            "interests": "Stocks, Pickleball, Boxing",
            "contact": "dtgordon@scu.edu",
        },
        {
            "name": "Dominick Kubica",
            "role": "University Researcher",
            "about": "Dominick is an aspiring home-grown data scientist with a "
                     "passion for finance and technology. ML and AI enthusiast.",
            "interests": "Data Science, Weightlifting, Cooking",
            "contact": "dominickkubica@gmail.com",
        },
        {
            "name": "Nanami Emura",
            "role": "University Researcher",
            "about": "Nanami developed the core sentiment analysis algorithm "
                     "and specializes in transformer models for financial text analysis.",
            "interests": "Deep Learning, Soccer, Photography",
            "contact": "nemura@scu.edu",
        },
        {
            "name": "Derleen Saini",
            "role": "University Researcher",
            "about": "Derleen created this Streamlit application and specializes in "
                     "data visualization and user-experience design.",
            "interests": "UI/UX Design, Photography, Yoga",
            "contact": "dsaini@scu.edu",
        },
        {
            "name": "Charles Goldenberg",
            "role": "Practicum Project Lead",
            "about": "Charles is the leader of our practicum project and has extensive "
                     "experience working with technical projects.",
            "interests": "Statistical Modeling, Travel, Jazz",
            "contact": "cgoldenberg@scu.edu",
        },
    ]

    # Add CSS for profile cards
    st.markdown("""
    <style>
    .profile-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .profile-card img {
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # layout: 3 + 2
    rows = [st.columns(3), st.columns(2)]
    idx = 0
    for cols in rows:
        for col in cols:
            if idx >= len(team_members):
                break
            member = team_members[idx]
            idx += 1

            with col:
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                
                display_profile_image(member["name"])

                st.markdown(
                    f"""
                    <h3>{member['name']}</h3>
                    <p style="color:{microsoft_colors['primary']};font-weight:bold;">
                        {member['role']}
                    </p>
                    <p>{member['about']}</p>
                    <p><strong>Interests:</strong> {member['interests']}</p>
                    <p><strong>Contact:</strong> {member['contact']}</p>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # contact form ------------------------------------------------------------
    st.markdown("## Contact Us")
    with st.form("contact_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.text_input("Name")
            st.text_input("Email")
            st.text_input("Subject")
        with c2:
            st.text_area("Message", height=150)

        if st.form_submit_button("Send Message"):
            st.success("Thanks! Your message has been sent.")

###############################################################################
#  Quick standalone test
###############################################################################
if __name__ == "__main__":
    st.set_page_config(page_title="About Us Demo", layout="wide")
    
    # Display setup instructions for first-time users
    if not (Path(__file__).parent / "assets").exists():
        st.warning("""
        **Image Setup Required**: 
        
        Please create an 'assets' folder in the same directory as this script and place your team member images there:
        - DYLAN.png
        - DOMINICK.png
        - NANAMI.png
        - DERLEEN.png
        - CHARLES.png
        """)
    
    render_about_us_tab({"primary": "#0078d4"})
