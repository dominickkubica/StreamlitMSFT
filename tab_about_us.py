import streamlit as st

def render_about_us_tab(microsoft_colors, display_profile_image):
    st.markdown("## Meet Our Team")
    st.markdown("We are a group of passionate data scientists and financial analysts working to revolutionize how earnings calls are analyzed.")
    
    # Team profiles in two rows
    row1_cols = st.columns(3)
    row2_cols = st.columns(2)
    
    team_members = [
        {
            "name": "Dylan Gordon",
            "role": "University Researcher",
            "about": "Dylan is a former chemical engineer turned data scientist. He specializes in optimization and machine learning.",
            "interests": "Stocks, Pickleball, Boxing",
            "contact": "dtgordon@scu.edu"
        },
        {
            "name": "Dominick Kubica",
            "role": "University Researcher",
            "about": "Dominick is an aspiring homegrown data scientist with a passion for finance and technology. ML and AI enthusiast.",
            "interests": "Data Science, Weightlifting, Cooking",
            "contact": "dominickkubica@gmail.com"
        },
        {
            "name": "Nanami Emura",
            "role": "University Researcher",
            "about": "Nanami developed the core sentiment analysis algorithm and specializes in transformer models for financial text analysis.",
            "interests": "Deep Learning, Soccer, Photography",
            "contact": "nemura@scu.edu"
        },
        {
            "name": "Derleen Saini",
            "role": "University Researcher",
            "about": "Derleen created this Streamlit application and specializes in data visualization and user experience design.",
            "interests": "UI/UX Design, Photography, Yoga",
            "contact": "dsaini@scu.edu"
        },
        {
            "name": "Charles Goldenberg",
            "role": "Practicum Project Lead",
            "about": "Charles is the leader of our Practicum project and has extensive experience working with Technical Projects",
            "interests": "Statistical Modeling, Travel, Jazz",
            "contact": "cgoldenberg@scu.edu"
        }
    ]
    
    # First row - 3 profiles
    for i, col in enumerate(row1_cols):
        if i < len(team_members):
            with col:
                # Profile container
                st.markdown(f'<div class="profile-card">', unsafe_allow_html=True)
                
                # Display profile image directly using st.image
                display_profile_image(team_members[i]['name'])
                
                # Display profile information
                st.markdown(f"""
                <h3>{team_members[i]['name']}</h3>
                <p style="color: {microsoft_colors['primary']}; font-weight: bold;">{team_members[i]['role']}</p>
                <p>{team_members[i]['about']}</p>
                <p><strong>Interests:</strong> {team_members[i]['interests']}</p>
                <p><strong>Contact:</strong> {team_members[i]['contact']}</p>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Second row - 2 profiles
    for i, col in enumerate(row2_cols):
        idx = i + 3
        if idx < len(team_members):
            with col:
                # Profile container
                st.markdown(f'<div class="profile-card">', unsafe_allow_html=True)
                
                # Display profile image directly using st.image
                display_profile_image(team_members[idx]['name'])
                
                # Display profile information
                st.markdown(f"""
                <h3>{team_members[idx]['name']}</h3>
                <p style="color: {microsoft_colors['primary']}; font-weight: bold;">{team_members[idx]['role']}</p>
                <p>{team_members[idx]['about']}</p>
                <p><strong>Interests:</strong> {team_members[idx]['interests']}</p>
                <p><strong>Contact:</strong> {team_members[idx]['contact']}</p>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact form
    st.markdown("## Contact Us")
    
    st.markdown("<div class='contact-form'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Name")
        st.text_input("Email")
        st.text_input("Subject")
        
    with col2:
        st.text_area("Message", height=150)
        
    st.button("Send Message", key="send_message_btn")
    st.markdown("</div>", unsafe_allow_html=True)
