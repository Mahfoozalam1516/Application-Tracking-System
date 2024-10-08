import streamlit as st
import re
from collections import Counter

def extract_text_from_pdf(pdf_file):
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        import docx2txt
        text = docx2txt.process(docx_file)
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_skills(text):
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
    
    for skill in skill_patterns:
        if skill in text:
            skills_found.append(skill)
    
    return list(set(skills_found))

def calculate_similarity(text1, text2):
    # Simple word overlap similarity
    words1 = set(preprocess_text(text1).split())
    words2 = set(preprocess_text(text2).split())
    
    common_words = words1.intersection(words2)
    unique_words = words1.union(words2)
    
    if not unique_words:
        return 0
    
    return len(common_words) / len(unique_words) * 100

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
            
            if resume_text:
                # Calculate match percentage
                match_percentage = calculate_similarity(job_description, resume_text)
                
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