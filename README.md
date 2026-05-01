# Computer Science Journal Finder

This project implements a journal recommendation system for computer science articles. Given an article abstract, the system recommends the top 5 most relevant academic journals.

The project also performs topic clustering on computer science publications using article abstracts, author keywords, Web of Science Keyword Plus terms, and subject categories.

## Project Overview

The main goals of this project are:

1. Build a journal finder system.
2. Recommend the top 5 relevant journals for a given article abstract.
3. Generate topic clusters for subject areas.

## Dataset

The SQLite database includes article metadata, abstracts, journals, keywords, Keyword Plus terms, and subject categories.

Main tables used:

- AcademicRecord
- AcademicRecordAbstract
- Publication
- AcademicRecordKeyword
- AcademicKeyword
- AcademicRecordKeywordPlus
- AcademicKeywordPlus
- AcademicRecordSubject
- AcademicSubject

## Methodology

### 1. Database Exploration

The database was explored using SQLite and pandas. The main dataset was created by joining article records, abstracts, publication information, keywords, Keyword Plus terms, and subject categories.

Notebook:

notebooks/01_database_exploration.ipynb

Generated file:

data/master_articles.csv

### 2. Journal Recommendation Model

A text classification approach was used for journal recommendation.

Pipeline:

Abstract text -> Text cleaning -> TF-IDF vectorization -> SGDClassifier

The model predicts probability scores for journals and returns the top 5 journals with the highest scores.

Notebook:

notebooks/02_journal_finder_model.ipynb

Saved model:

models/journal_recommender_tfidf_sgd.pkl

Model performance:

Top-1 Accuracy: 0.3623  
Top-5 Accuracy: 0.6616

### 3. Topic Clustering

Topic clustering was performed using article text, keywords, Keyword Plus terms, and subject categories.

Pipeline:

Text fields -> TF-IDF vectorization -> TruncatedSVD -> KMeans clustering

Notebook:

notebooks/03_topic_clustering.ipynb

Generated files:

- data/clustered_articles.csv
- data/cluster_summary.csv
- data/cluster_representative_articles.csv

The final clustering model uses 10 interpretable topic clusters.

## How to Run

### 1. Create virtual environment

python3 -m venv .venv  
source .venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run journal recommendation from terminal

Example:

python src/recommend.py --text "This paper proposes a machine learning approach for information retrieval and text classification using natural language processing techniques."

Example output:

Top Journal Recommendations

1. INFORMATION RETRIEVAL
2. FOUNDATIONS AND TRENDS IN INFORMATION RETRIEVAL
3. JOURNAL OF INTELLIGENT INFORMATION SYSTEMS
4. JOURNAL OF ARTIFICIAL INTELLIGENCE RESEARCH
5. LANGUAGE RESOURCES AND EVALUATION

### 4. Run with abstract from a text file

python src/recommend.py --file sample_abstract.txt

## Project Structure

journal-finder-project/

- data/
- models/
- notebooks/
- src/
- report/
- requirements.txt
- README.md
- .gitignore

## Technologies Used

- Python
- SQLite
- pandas
- scikit-learn
- TF-IDF
- SGDClassifier
- TruncatedSVD
- KMeans
- Jupyter Notebook


## Important Note About Large Files

Large database, generated dataset, and trained model files are not included in this GitHub repository because of file size limitations.

To reproduce the project:

1. Place `CompSciencePub.sqlite` inside the `data/` folder.
2. Run `notebooks/01_database_exploration.ipynb`.
3. Run `notebooks/02_journal_finder_model.ipynb` to generate the trained journal recommendation model.
4. Run `notebooks/03_topic_clustering.ipynb` to generate topic clustering outputs.

After running the notebooks, the terminal recommendation script can be used with:

`python src/recommend.py --text "your abstract text here"`

## Author

Alp Tekin Kahveci