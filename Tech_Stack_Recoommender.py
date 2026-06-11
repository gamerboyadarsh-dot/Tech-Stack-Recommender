import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def tech_recommender():
    print("=========================================================")
    print(" DIGITAL MATCHMAKER: TECH STACK RECOMMENDER INITIALIZED ")
    print("=========================================================\n")
    # --- 1. INGESTION PHASE ---
    # Simulating the raw_skills.csv dataset with various career paths
    roles_data ={
        "Job Role":["DevOps Engineer", 
            "Data Scientist", 
            "Backend Developer", 
            "Technical Entry Scheme (Armed Forces)", 
            "Cloud Architect"],"Skills" :["aws docker kubernetes linux python git",
            "python r sql machine learning tensors optimization data analysis",
            "java python sql apis system design spring",
            "functional strength leadership java c networking security communication",
            "aws azure cloud computing networking python architecture"]
    }
    df = pd.DataFrame(roles_data)
    # Ingestion: The script must accept a minimum of three user inputs
    print("Please enter at least 3 skills, interests, or career goals.")
    print("Example: 'python cloud automation'")
    # Pre-filled with an example to test the engine immediately
    user_skills = input("You: ") or "java c networking leadership"
    # --- 2. SCORING PHASE ---
    # Combine user profile with the item dataset to build a shared vocabulary space
    all_text = df['Skills'].tolist() + [user_skills]
    # TF-IDF penalizes generic words and rewards highly specific, descriptive tags
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    # The user vector is the very last item in our new matrix
    user_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]
    # Calculate Cosine Similarity (measuring the mathematical angle between vectors)
    similarity_scores = cosine_similarity(user_vector, job_vectors)[0]
    # --- 3. SORTING PHASE ---
    # Add the calculated scores to our dataframe
    df['Match Score'] = similarity_scores

    # Organize the scored dataset in descending order
    sorted_df = df.sort_values(by='Match Score', ascending=False)

    # --- 4. FILTERING PHASE ---
    # Prevent choice overload by truncating the output to a "Top-N" list
    top_3_matches = sorted_df.head(3)

    print("\n=========================================================")
    print(" TOP 3 RECOMMENDED CAREER PATHS")
    print("=========================================================")
    
    # Iterate through the results and format the output
    for index, row in top_3_matches.iterrows():
        # Formatting to show the match as a clean percentage
        print(f"[{row['Match Score']*100:05.2f}% Match] | {row['Job Role']}")

if __name__ == "__main__":
    tech_recommender()