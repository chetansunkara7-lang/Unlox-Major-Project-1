import streamlit as st
import pandas as pd
import re
import io
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + " "
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(file):
    text = ""
    try:
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
    return text

def clean_text(text):
    text = text.lower()
    #Replace anything that isn't a letter, number, space, +, #, or . (c++, c#, .net) with a space
    text = re.sub(r'[^a-z0-9\s\+#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skill_list):
    found_skills = []
    #Keyword matching
    for skill in skill_list:
        #Using word boundaries to avoid partial matches (e.g., finding "c" in "machine")
        #Escaping regex characters in skills like c++
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    return found_skills

def generate_roadmap(missing_skills):
    roadmap = {}
    for i, skill in enumerate(missing_skills):
        roadmap[f"Week {i+1}"] = f"Learn basics and practical applications of {skill.title()}"
    return roadmap

def main():
    st.title("AI Resume Analyzer & Job Recommender")
    st.write("Upload your resume to see how well it matches various job roles.")

    try:
        roles_df = pd.read_csv('data/job_roles.csv')
        skills_df = pd.read_csv('data/skill_dictionary.csv')
        skill_dictionary = skills_df['Skill'].tolist()
    except FileNotFoundError:
        st.error("Datasets not found. Please run the data generation script first.")
        return

    #Resume Upload
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'])

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        #Extract text based on file type
        if uploaded_file.name.endswith('.pdf'):
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = extract_text_from_docx(uploaded_file)

        cleaned_text = clean_text(raw_text)

        #Extraction of skills from resume
        resume_skills = extract_skills(cleaned_text, skill_dictionary)

        st.subheader("Extracted Skills")
        st.write(", ".join([s.title() for s in resume_skills]) if resume_skills else "No recognizable skills found.")

        #Target role selection
        target_role = st.selectbox("Select Target Job Role", roles_df['Job Role'].tolist())

        if st.button("Analyze Match"):
            role_data = roles_df[roles_df['Job Role'] == target_role].iloc[0]
            required_skills_str = role_data['Expected Skills']
            required_skills_list = [s.strip() for s in required_skills_str.split(',')]

            vectorizer = TfidfVectorizer()
            resume_skills_str = " ".join(resume_skills)
            vectors = vectorizer.fit_transform([resume_skills_str, required_skills_str])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            match_score = round(similarity * 100)

            missing_skills = [skill for skill in required_skills_list if skill not in resume_skills]

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Match Score: {match_score}%")
                fig = px.pie(
                    values=[match_score, 100 - match_score],
                    names=['Matched', 'Missing'],
                    hole=0.7,
                    color_discrete_sequence=['#4CAF50', '#FF5252']
                )
                fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Missing Skills:")
                if missing_skills:
                    for skill in missing_skills:
                        st.write(f"- {skill.title()}")
                else:
                    st.success("You have all the core required skills!")

            st.subheader("Suggested Learning Roadmap")
            if missing_skills:
                roadmap = generate_roadmap(missing_skills)
                for week, task in roadmap.items():
                    st.info(f"**{week}:** {task}")
            else:
                st.write("Your skill profile looks great for this role. Focus on advanced projects!")

            st.divider()
            st.subheader("Other Recommended Roles")
            recommendations = []
            for idx, row in roles_df.iterrows():
                if row['Job Role'] != target_role:
                    role_req_str = row['Expected Skills']
                    vecs = vectorizer.fit_transform([resume_skills_str, role_req_str])
                    sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
                    recommendations.append((row['Job Role'], round(sim * 100)))

            #Sort by highest score
            recommendations.sort(key=lambda x: x[1], reverse=True)
            for i, (role, score) in enumerate(recommendations[:3]): # Top 3
                st.write(f"{i+1}. **{role}**: {score}% Match")

if __name__ == "__main__":
    main()
