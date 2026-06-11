# 🎯 Tech Stack Recommender (Digital Matchmaker)

An AI-driven recommendation engine built in Python that maps a user's raw skills to their ideal tech career paths. 

I built this project to transition away from basic heuristic programming (hard-coded `if/else` rules) and dive into **Active Prediction**. Instead of explicitly telling the machine what to do, this engine uses mathematical vectors to understand user intent and cure "choice overload" when deciding what technologies to learn next.

## 🧠 How the Brain Works (The Math)

This isn't a collaborative filter (it doesn't care what "other users" clicked on). It uses **Content-Based Filtering** to match the intrinsic properties of job roles directly to the user's profile.

Here is how the engine processes data under the hood:

1. **TF-IDF Weighting (Term Frequency-Inverse Document Frequency):** A simple binary overlap (1 for yes, 0 for no) is flawed because it treats generic words the same as highly specific tech skills. I implemented TF-IDF to penalize high-frequency, generic terms and heavily reward specific, descriptive tags.
2. **Vector Mapping:** The engine translates qualitative concepts (like "Java", "Cloud", "Algorithms") into quantitative numerical arrays (vectors) within a shared vocabulary space.
3. **Cosine Similarity:** Instead of measuring the straight-line Euclidean distance (which fails when comparing a massive job description to a short 3-word user prompt), the algorithm measures the *mathematical angle* between the vectors. 
   * `1.0` = Perfect alignment.
   * `0.0` = Completely orthogonal (no shared traits).

## ⚙️ The Pipeline Architecture

The system strictly follows a 4-step execution pipeline:
* **Ingestion:** Captures a minimum of 3 user skills/interests to bypass the "User Cold Start" problem.
* **Scoring:** Calculates the Cosine Similarity score between the user's vector and every job role vector in the dataset.
* **Sorting:** Organizes the dataset in descending order based on the angular alignment.
* **Filtering:** Truncates the output to generate a clean, focused **Top 3** list to prevent choice overload.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)

## 🚀 How to Run It Locally

Want to test it out and see what tech stack you should learn next?

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/gamerboyadarsh-dot/tech-stack-recommender.git](https://github.com/gamerboyadarsh-dot/tech-stack-recommender.git)
