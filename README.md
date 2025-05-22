# CodeAlpha_Music_Recommendation
Machine Learning Internship Task 1 - CodeAlpha
# 🎧 Music Recommendation System

This project is part of my Machine Learning Internship at **CodeAlpha**.  
The goal was to build a system that can predict whether a user is likely to replay a song based on audio features and metadata.

---

## 📌 Objective

To develop a **binary classification model** that predicts the probability of a song being **replayed** or **not replayed**, using song-level features like energy, danceability, tempo, etc.

---

## 🧠 Technologies Used

- Python
- Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn
- Random Forest Classifier
- Logistic Regression (with class balancing)

---

## 📁 Dataset

The dataset includes 170,000+ rows of songs with features:
- `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `tempo`, `duration`, etc.
- `popularity` (used to create binary label)

📌 Binary label:  
`1` → Song is likely to be replayed (popularity ≥ 50)  
`0` → Otherwise

---

## 🧪 ML Pipeline

1. **Preprocessing:**
   - Removed unnecessary columns (`id`, `name`, `release_date`, etc.)
   - Converted `popularity` to binary `replayed` label

2. **Model Training:**
   - Logistic Regression (class_weight='balanced')
   - Random Forest (on 20k sampled dataset)

3. **Evaluation:**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix Visualizations

---

## 📊 Results

| Model               | Accuracy | Recall (Replayed) |
|---------------------|----------|-------------------|
| Logistic Regression | 66.6%    | 76.7% ✅          |
| Random Forest       | ~75%     | Balanced ⚖️       |

---

## 🧠 Learnings

- Handling class imbalance with `class_weight='balanced'`
- Converting continuous features into categorical targets
- Training on real-world large-scale data
- Evaluating ML models using visual tools

---

## 📬 Submission

✅ GitHub Link: [Paste Your Repo Link Here]  
📽️ Shared on LinkedIn: [Your LinkedIn Post Link]

---

## 🔗 Credits

This project was completed as part of the **Machine Learning Internship** by [CodeAlpha](https://www.codealpha.tech)

---

## 🧑‍💻 Author

**Your Name**  
ML Intern @ CodeAlpha  
LinkedIn: [Your LinkedIn Profile]
