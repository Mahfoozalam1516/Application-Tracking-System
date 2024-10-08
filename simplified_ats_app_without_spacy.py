import streamlit as st
import pandas as pd
import numpy as np
import PyPDF2
import docx2txt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file):
    text = docx2txt.process(docx_file)
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_skills(text):
    # Common skill keywords (you can expand this list)
    skill_patterns = [
        'python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 
        'angular', 'node', 'django', 'flask', 'machine learning', 
        'data analysis', 'project management', 'agile', 'aws', 'docker',
        'kubernetes', 'devops', 'ci/cd', 'git', 'linux', 'excel',
        'tableau', 'power bi', 'statistics', 'hadoop', 'spark',
        'tensorflow', 'pytorch', 'nlp', 'computer vision'
    ]
    
    text = text.lower()
    skills_found = []
    
    # Find all matching skills
    for skill in skill_patterns:
        if skill in text:
            skills_found.append(skill)
    
    return list(set(skills_found))

def calculate_match_percentage(job_description, resume_text):
    # Preprocess both texts
    processed_jd = preprocess_text(job_description)
    processed_resume = preprocess_text(resume_text)
    
    # Create document vectors
    vectorizer = CountVectorizer()
    doc_vectors = vectorizer.fit_transform([processed_jd, processed_resume])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(doc_vectors[0:1], doc_vectors[1:2])[0][0]
    
    return similarity * 100

def main():
    st.title("Resume Analyzer - ATS System")
    st.write("Upload a resume and job description to analyze the match percentage")
    
    # Job Description input
    job_description = st.text_area("Enter the Job Description", height=200)
    
    # Resume upload
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf', 'docx'])
    
    if resume_file is not None and job_description:
        try:
            # Extract text from resume
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = extract_text_from_docx(resume_file)
            
            # Calculate match percentage
            match_percentage = calculate_match_percentage(job_description, resume_text)
            
            # Extract skills
            skills = extract_skills(resume_text)
            
            # Display results
            st.header("Analysis Results")
            
            # Create three columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Match Percentage", f"{match_percentage:.1f}%")
            
            with col2:
                st.metric("Skills Found", len(skills))
            
            with col3:
                status = "High" if match_percentage > 70 else "Medium" if match_percentage > 50 else "Low"
                st.metric("Match Level", status)
            
            # Display skills
            st.subheader("Skills Identified")
            st.write(", ".join(skills))
            
            # Detailed Analysis
            st.subheader("Detailed Analysis")
            
            # Create a more detailed breakdown
            analysis_df = pd.DataFrame({
                'Category': ['Overall Match', 'Skills Match', 'Keywords Match'],
                'Score': [
                    f"{match_percentage:.1f}%",
                    f"{min(len(skills) * 10, 100)}%",
                    f"{match_percentage * 0.8:.1f}%"
                ]
            })
            
            st.table(analysis_df)
            
            # Suggestions
            st.subheader("Suggestions for Improvement")
            if match_percentage < 70:
                st.write("Here are some suggestions to improve your resume:")
                suggestions = [
                    "Add more relevant keywords from the job description",
                    "Highlight specific skills mentioned in the job posting",
                    "Quantify your achievements with metrics",
                    "Use action verbs to describe your experience"
                ]
                for suggestion in suggestions:
                    st.write(f"• {suggestion}")
            else:
                st.write("Your resume appears to be well-matched with the job description!")
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")

if __name__ == "__main__":
    main()