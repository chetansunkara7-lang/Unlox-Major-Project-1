AI Resume Analyzer & Job Recommender
This project is an NLP-based application that evaluates a candidate's resume against specific job roles to determine a match score and identify skill gaps.
Features
Extracts text from PDF and DOCX files.
Cleans and normalizes unstructured text while preserving technical keywords.
Uses TF-IDF and Cosine Similarity to calculate a match score.
Provides a gap analysis and generates a weekly learning roadmap for missing skills.
How to Run
Install dependencies: pip install -r requirements.txt
Run the application: streamlit run app.py
Upload a resume and select a target role to view the analysis.
