Representation Geometry in SVD-Based Collaborative Filtering

Analyzing How Data Sparsity Distorts Embedding Stability

Overview

This project demonstrates a critical failure mode in collaborative filtering systems: users with sparse interaction histories receive unreliable recommendations despite good aggregate performance metrics. Using SVD-based matrix factorization on the MovieLens 1M dataset, I quantify how representation stability degrades with data sparsity.

Analysis Question

How does data availability affect the stability and reliability of learned representations in recommender systems?

Key Findings

• Sparse users (<10 ratings) show ~50% median embedding shift across retraining
• Dense users (≥50 ratings) remain stable at ~5% shift
• This represents 10x more instability for sparse users
• Standard RMSE metrics mask this systematic failure
• 40% of users are affected by this reliability gap

Dataset

MovieLens 1M
• 1 million ratings from 6,000 users on 4,000 movies
• Publicly available from GroupLens
• URL: https://grouplens.org/datasets/movielens/1m/
• Download: ml-1m.zip (~6MB)

Quick Start

1. Install dependencies:
pip install numpy pandas scipy scikit-learn matplotlib seaborn
2. Run the analysis:
python svd_representation_analysis.py
The script will automatically download the MovieLens 1M dataset if not present, perform the stability analysis, and generate visualizations.

Files in This Repository

• svd_representation_analysis.py - Main analysis code
• svd_representation_geometry.pptx - Presentation slides
• narration_script.docx - Speaking notes for presentation
• code_walkthrough_narration.docx - Detailed code explanation
• README.docx - This document

Methodology

1. Train multiple SVD models
Train 5 independent SVD models (50 latent factors) with different random initializations
2. Measure embedding stability
Compute cosine distance between user embeddings across all model pairs
3. Analyze by user category
• Sparse: <10 ratings
• Medium: 10-50 ratings
• Dense: ≥50 ratings

Results
Embedding Stability (Cosine Distance):
• Sparse users: Mean = 0.506, Median = 0.503
• Medium users: Mean = 0.152, Median = 0.148
• Dense users: Mean = 0.051, Median = 0.049
Standard RMSE Performance:
• Overall: 0.82
• Sparse users: 0.89
• Dense users: 0.78

Implications

Evaluation Gap
Standard evaluation metrics can mask systematic failures for underrepresented populations. A model with acceptable RMSE may still provide unreliable recommendations to 40% of users.

Reliability Threshold
Data availability creates a reliability threshold around 10 interactions. Users below this threshold receive fundamentally unreliable recommendations regardless of aggregate performance.

Diagnostic Approach
Representation-based diagnostics can reveal failures invisible to standard metrics, enabling more rigorous model evaluation and fairer system deployment.

Technical Details

• Algorithm: Truncated SVD via scipy.sparse.linalg.svds
• Latent factors: 50
• Stability metric: Cosine distance (1 - cosine similarity)
• Evaluation: 80/20 train/test split
• Visualizations: Matplotlib and Seaborn
Future Extensions
• Compare with other matrix factorization techniques (NMF, ALS)
• Analyze embedding stability in neural collaborative filtering
• Investigate robustness to different sparsity patterns
• Develop stability-aware training objectives
• Test on other domains (e-commerce, news, social networks)

Author

Gulusan Erdogan-Ozgul
Portfolio Project - ML Reliability Analysis
License
This project is open source and available for educational purposes.
Acknowledgments
• Dataset: MovieLens 1M from GroupLens
• Inspired by real-world challenges in deploying reliable recommender systems
• Created as part of portfolio demonstrating ML reliability analysis
