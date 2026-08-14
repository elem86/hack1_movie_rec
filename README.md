# AI-Powered Movie Recommendation System

A Python movie recommendation project built for a data-analysis hackathon. It
uses the MovieLens small dataset, enriches the catalog with TMDB metadata, and
combines several recommendation strategies to produce personalized movie
suggestions and human-readable explanations.

The project demonstrates Python fundamentals, Pandas and NumPy data processing,
Matplotlib and Seaborn visualization, SciPy statistics and clustering, functional
programming, and object-oriented design.

## Features

- Content-based recommendations from user genre and director preferences.
- User-based collaborative filtering with SciPy Pearson correlation.
- User segmentation with SciPy k-means clustering.
- A hybrid ranker combining content, collaborative, and cluster signals.
- Exclusion of movies the selected user has already rated.
- Rule-based explanations such as “Since you liked ..., you might enjoy ...”.
- Chronological train/test analysis with a paired Wilcoxon signed-rank test.
- Genre, rating, popularity, sparsity, cluster, and user-history visualizations.
- An interactive `ipywidgets` demo for existing and new users.

## Recommendation approaches

### Content-based filtering

The content model scores unwatched movies using the selected user's historical
genre ratings, director ratings, and the movie's global MovieLens average. A
minimum popularity threshold reduces the influence of movies with very few
ratings.

### Collaborative filtering

The collaborative model calculates Pearson correlation between the target user
and users who rated enough of the same movies. Ratings from positively correlated
neighbors are similarity-weighted to rank unseen movies.

### K-means clustering

Users are represented by genre ratings relative to their own average rating.
SciPy k-means groups similar profiles, and highly rated unseen movies from the
target user's cluster become recommendation candidates.

### Hybrid recommendations

The three model scores are converted to percentile ranks and combined using the
following default weights:

- Content-based: 40%
- Collaborative: 40%
- Cluster-based: 20%

`models_recommending` records how many of the three systems included a movie in
their candidate lists. It is a measure of model agreement, not a predicted
rating.

## Project structure

```text
hack1_movie_rec/
|-- data/
|   |-- raw/                       # Original MovieLens CSV files
|   `-- processed/
|       `-- tmdb_movies.csv        # Cached TMDB enrichment
|-- notebooks/
|   |-- data/processed/
|   |   |-- movie_ratings.csv      # Enriched rating-level dataset
|   |   `-- user_clusters.csv      # Saved k-means cluster assignments
|   |-- 01_data_exploration.ipynb
|   |-- 02_recommendation_system.ipynb
|   |-- 03_final_demo.ipynb
|   `-- recommender.py             # Reusable recommendation classes
|-- README.md
`-- requirements.txt
```

## Setup

Python 3.10 or newer is recommended. The project was validated with Python
3.13.

From the project root on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## TMDB authentication

The included `data/processed/tmdb_movies.csv` cache means a TMDB token is not
needed when all required movie metadata is already present. If the cache is
missing or incomplete, notebook 01 asks for a TMDB API Read Access Token.

You may enter the token securely when prompted or set it before starting
Jupyter:

```powershell
$env:TMDB_TOKEN = "your-read-access-token"
```

```bash
export TMDB_TOKEN="your-read-access-token"
```

Do not commit an API token to the repository.

## Running the notebooks

Start Jupyter from the `notebooks` directory. This is important because the
processed-data paths in notebooks 02 and 03 are relative to that directory.

```powershell
cd notebooks
jupyter lab
```

Run the notebooks in order:

1. `01_data_exploration.ipynb` loads and enriches the data, performs EDA, and
   produces `data/processed/movie_ratings.csv`.
2. `02_recommendation_system.ipynb` builds and analyzes the recommendation
   models and produces `data/processed/user_clusters.csv`.
3. `03_final_demo.ipynb` loads the processed artifacts and launches the
   existing-user and new-user interactive demos.

If the processed CSV files are already present, you can open notebook 03
directly.

### Path note

Notebook 01 currently contains absolute Windows paths in its first data-loading
cell and in `TMDB_CACHE`. If the repository is moved to another location, update
those paths before running notebook 01. Notebooks 02 and 03 use paths relative to
the `notebooks` directory.

## Using the demo

The final notebook provides two tabs:

- **Demo Mode:** select an existing MovieLens user ID and request 3–10 movies.
- **Try It Yourself:** search for movies, rate at least five, and generate a
  temporary preference profile and recommendations.

The new-user profile exists only in memory and does not modify the original
MovieLens data.

## Statistical analysis

The project uses an 80/20 chronological split for each user. Genre preferences
are inferred from the earlier 80% of ratings and compared with ratings in the
later 20%. A one-sided paired Wilcoxon signed-rank test evaluates whether users
rate preference-matching movies more highly.

This test supports the usefulness of the genre signal, but it is not a complete
evaluation of recommendation ranking quality.

## Known limitations and useful next steps

- MovieLens timestamps indicate when a rating was recorded, not necessarily
  when a movie was watched.
- The MovieLens small dataset does not contain names or ages, so profiles are
  identified by user ID and preferences are inferred from ratings.
- Genre and director scores could use Bayesian shrinkage so a single rating has
  less influence.
- New users should ideally be transformed with the exact feature scaling used
  to train k-means.
- The hybrid engine could add explicit fallbacks when one model returns no
  candidates for a sparse user.
- A production evaluation should report temporal Precision@K, Recall@K, hit
  rate, or NDCG in addition to the statistical genre analysis.
- The generated explanations are deterministic templates; no external language
  model is used.

## Data and acknowledgements

- Ratings and movie identifiers come from the
  [MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/).
- Additional movie metadata comes from
  [The Movie Database (TMDB)](https://www.themoviedb.org/).

This product uses the TMDB API but is not endorsed or certified by TMDB.
