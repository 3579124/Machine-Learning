# AIDAMS ML Competition - Kaggle Solution Guide

This comprehensive guide will help you use the `kaggle_solution.ipynb` notebook to build a winning solution for the AIDAMS ML competition.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Options](#configuration-options)
3. [Model Comparison](#model-comparison)
4. [Tips for Better Performance](#tips-for-better-performance)
5. [Understanding the Output](#understanding-the-output)
6. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Step 1: Installation

First, install all required dependencies:

```bash
pip install -r requirements_enhanced.txt
```

**Note:** If you encounter any installation issues, try installing packages individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm catboost
pip install joblib jupyter ipykernel
```

### Step 2: Prepare Your Data

Place your Kaggle competition data files in the same directory as the notebook:

- `data.csv` - Training data with the `Target` column
- `test.csv` - Test data without the `Target` column (optional, for generating submissions)

**File Format:** The notebook automatically handles both comma and semicolon delimiters, so you don't need to worry about CSV format.

### Step 3: Run the Notebook

You can run the notebook in two ways:

**Option A: Jupyter Notebook (Local)**
```bash
jupyter notebook kaggle_solution.ipynb
```

**Option B: Kaggle Platform**
1. Go to Kaggle.com and create a new notebook
2. Upload the `kaggle_solution.ipynb` file
3. Add the competition dataset
4. Run all cells

### Step 4: Submit to Kaggle

After running the notebook, you'll find a submission file:
- `submission_xgboost.csv` (if using XGBoost)
- `submission_lightgbm.csv` (if using LightGBM)
- `submission_catboost.csv` (if using CatBoost)

Upload this file to the Kaggle competition submission page.

---

## ⚙️ Configuration Options

The notebook has several configuration options at the top (Section 1). Here's how to use them:

### Model Selection

Choose which model to train and use for predictions:

```python
SELECTED_MODEL = 'xgboost'  # Options: 'xgboost', 'lightgbm', 'catboost'
```

**When to use each model:**
- **XGBoost**: Great all-around performance, widely used in competitions
- **LightGBM**: Faster training, good for large datasets
- **CatBoost**: Best for datasets with many categorical features

### Hyperparameter Tuning

Enable/disable automated hyperparameter tuning:

```python
ENABLE_HYPERPARAMETER_TUNING = False  # Set to True to enable
```

⚠️ **Warning:** Hyperparameter tuning can take 30-60 minutes or more! Only enable this when you have time and want to squeeze out extra performance.

**What it does:**
- Searches through different combinations of parameters
- Uses RandomizedSearchCV with 20 iterations
- Performs 3-fold cross-validation for each combination
- Automatically selects the best parameters

### Feature Engineering

Enable/disable feature engineering:

```python
ENABLE_FEATURE_ENGINEERING = False  # Set to True to enable
```

**What it does:**
- Creates interaction features (e.g., Age × Grade)
- Generates polynomial features
- Adds domain-specific features

**Note:** Currently a placeholder. You can add your own feature engineering logic in Section 5.

### Visualizations

Enable/disable plots and charts:

```python
ENABLE_VISUALIZATIONS = True  # Set to False to disable
```

**What it includes:**
- Target variable distribution bar plot
- Correlation heatmap for numerical features

**When to disable:** On Kaggle platform or when running in headless mode

### Other Settings

```python
RANDOM_SEED = 42           # For reproducibility
TEST_SIZE = 0.2            # Validation set size (20%)
CV_FOLDS = 5               # Number of cross-validation folds
TRAIN_DATA_PATH = 'data.csv'
TEST_DATA_PATH = 'test.csv'
```

---

## 📊 Model Comparison

### Running All Three Models

To compare all three models and choose the best one:

1. **Run the notebook once** - This trains all three models (XGBoost, LightGBM, CatBoost)
2. **Check Section 10** - View the model comparison table
3. **Look at validation accuracy** - Higher is better
4. **Check overfitting** - Lower gap between training and validation is better
5. **Review CV scores in Section 11** - Most reliable metric

### Example Comparison Output

```
Model      Training Accuracy  Validation Accuracy  Overfitting
CatBoost          0.8523            0.8145           0.0378
XGBoost           0.8456            0.8123           0.0333
LightGBM          0.8489            0.8089           0.0400
```

### Choosing the Best Model

The notebook automatically identifies the best model based on validation accuracy. However, you should also consider:

1. **Cross-validation scores** (Section 11) - Most reliable metric
2. **Overfitting** - Lower is better (< 0.05 is good)
3. **Training time** - LightGBM is usually fastest
4. **Ensemble potential** - If models are close, consider averaging their predictions

### How to Switch Models

To use a different model for your final submission:

1. **Change the configuration** in Section 1:
   ```python
   SELECTED_MODEL = 'catboost'  # or 'xgboost' or 'lightgbm'
   ```

2. **Re-run Sections 11-15**:
   - Section 11: Cross-validation
   - Section 12: Hyperparameter tuning (optional)
   - Section 13: Final model training
   - Section 14: Generate submission
   - Section 15: Summary

You don't need to re-run Sections 1-10 unless you want to retrain all models.

---

## 💡 Tips for Better Performance

### 1. Try All Three Models

Each model has different strengths:
- Run the notebook to train all three
- Compare their CV scores (Section 11)
- Choose the one with the best CV score
- Consider ensembling if scores are close

### 2. Enable Hyperparameter Tuning

For a 1-2% accuracy boost:
```python
ENABLE_HYPERPARAMETER_TUNING = True
```

**Best practices:**
- Start with default parameters first
- Only tune after you've established a baseline
- Be patient - it takes time but often improves results

### 3. Feature Engineering

Add domain knowledge to create better features:

**Examples for student data:**
- Ratio features: `approved/enrolled` for each semester
- Aggregate features: `total_approved = sem1_approved + sem2_approved`
- Interaction features: `age × admission_grade`

Edit the `engineer_features()` function in Section 5 to add your ideas.

### 4. Ensemble Methods

Combine predictions from multiple models:

```python
# After training all three models (in a new cell)
ensemble_pred = (
    xgb_model.predict_proba(X_test) * 0.4 +
    lgb_model.predict_proba(X_test) * 0.3 +
    catboost_model.predict_proba(X_test) * 0.3
)
final_pred = np.argmax(ensemble_pred, axis=1)
```

Weight the models based on their CV scores.

### 5. Trust Cross-Validation Scores

**Why CV scores matter:**
- More reliable than single validation split
- Better estimate of real-world performance
- Less affected by lucky/unlucky splits

**Rule of thumb:**
- Validation accuracy can vary by ±2-3%
- CV accuracy is more stable
- If CV score is good but validation is low, trust CV

### 6. Data Quality Checks

Before training:
- Check for data leakage (features that shouldn't exist at prediction time)
- Verify no test information leaked into training
- Ensure consistent preprocessing between train and test

### 7. Iteration Strategy

**First iteration (quick baseline):**
- Use default parameters
- No hyperparameter tuning
- No feature engineering
- Get a submission score

**Second iteration (optimize):**
- Try all three models
- Enable hyperparameter tuning for the best model
- Add simple feature engineering

**Third iteration (fine-tune):**
- Create ensemble
- Advanced feature engineering
- Error analysis

---

## 🔍 Understanding the Output

The notebook uses visual symbols to make output easier to read:

### Symbols and Their Meanings

| Symbol | Meaning | Example |
|--------|---------|---------|
| ✓ | Success / Completed | `✓ Data loaded successfully` |
| ⚠️ | Warning | `⚠️ Missing values found` |
| ⊗ | Disabled / Skipped | `⊗ Hyperparameter tuning disabled` |
| 📊 | Data / Statistics | `📊 Dataset Information` |
| 🔍 | Analysis / Investigation | `🔍 Preprocessing training data...` |
| 🎯 | Target / Goal | `🎯 Target Variable Distribution` |
| 📈 | Metrics / Performance | `📈 Basic Statistical Summary` |
| 🚀 | Training / Execution | `🚀 Training XGBoost model...` |
| 🏆 | Best / Winner | `🏆 Best model: XGBoost` |
| 💾 | Saved / Stored | `💾 Saved Files` |
| 📤 | Output / Export | `📤 Generating submission file...` |
| 💡 | Tips / Suggestions | `💡 Tips for Better Performance` |

### Section-by-Section Output

**Section 2: Data Loading & EDA**
```
✓ Data loaded successfully
  Dataset shape: (4424, 37)
  Number of samples: 4424
  Number of features: 36
```
This tells you the data was loaded correctly and shows its dimensions.

**Section 7-9: Model Training**
```
✓ XGBoost training completed
  Training accuracy: 0.8523
  Validation accuracy: 0.8145
```
Higher validation accuracy is better. Training accuracy should be higher than validation (slight overfitting is normal).

**Section 10: Model Comparison**
```
Model      Training Accuracy  Validation Accuracy  Overfitting
CatBoost          0.8523            0.8145           0.0378
```
Lower overfitting value means better generalization.

**Section 11: Cross-Validation**
```
Mean CV Accuracy: 0.8123
Std CV Accuracy: 0.0234
Mean ± Std: 0.8123 ± 0.0234
```
- Mean: Average performance across all folds
- Std: Consistency (lower is better, means stable performance)

**Section 14: Submission**
```
Prediction distribution:
  Graduate: 2234 (50.5%)
  Dropout: 1876 (42.4%)
  Enrolled: 314 (7.1%)
```
Check if this distribution seems reasonable compared to your training data.

---

## 🔧 Troubleshooting

### Problem: FileNotFoundError: 'data.csv'

**Cause:** The training data file is not in the correct location.

**Solutions:**
1. Make sure `data.csv` is in the same directory as the notebook
2. Or change the path in Section 1:
   ```python
   TRAIN_DATA_PATH = '/path/to/your/data.csv'
   ```

### Problem: MemoryError during training

**Cause:** Not enough RAM to train the model.

**Solutions:**
1. **Reduce data size:**
   ```python
   # Add after loading data
   df = df.sample(frac=0.8, random_state=42)  # Use 80% of data
   ```

2. **Use smaller model parameters:**
   ```python
   # In model training cells
   n_estimators=100  # Instead of 500
   max_depth=4       # Instead of 6
   ```

3. **Disable hyperparameter tuning:**
   ```python
   ENABLE_HYPERPARAMETER_TUNING = False
   ```

4. **Use LightGBM instead** (more memory efficient):
   ```python
   SELECTED_MODEL = 'lightgbm'
   ```

### Problem: Low Validation Accuracy (< 0.70)

**Possible Causes and Solutions:**

1. **Data quality issues:**
   - Check for missing values (Section 2)
   - Verify target distribution is balanced
   - Look for outliers

2. **Model not trained long enough:**
   ```python
   # Increase iterations
   n_estimators=1000  # Instead of 500
   ```

3. **Need hyperparameter tuning:**
   ```python
   ENABLE_HYPERPARAMETER_TUNING = True
   ```

4. **Try a different model:**
   - If XGBoost gives 0.70, try CatBoost or LightGBM
   - They might work better for your specific data

5. **Add feature engineering:**
   ```python
   ENABLE_FEATURE_ENGINEERING = True
   # Then edit the engineer_features() function
   ```

### Problem: Submission File Format Error

**Cause:** Kaggle expects specific column names or format.

**Solutions:**

1. **Check required format** on Kaggle competition page

2. **Verify your submission file** has these columns:
   ```python
   # In Section 14, verify:
   print(submission_df.columns)  # Should show: ['id', 'Target']
   print(submission_df.head())
   ```

3. **Fix column names if needed:**
   ```python
   # Add this before saving submission
   submission_df.columns = ['id', 'Target']  # Match Kaggle format exactly
   ```

4. **Check for missing predictions:**
   ```python
   print(submission_df.isnull().sum())  # Should be 0 for all columns
   ```

### Problem: ModuleNotFoundError for a library

**Cause:** Required package not installed.

**Solution:**
```bash
# Install the specific package
pip install [package-name]

# Or reinstall all requirements
pip install -r requirements_enhanced.txt

# If using Conda
conda install [package-name]
```

### Problem: Kernel Dying / Notebook Crashing

**Causes:**
- Memory overflow
- Infinite loop
- System resource exhaustion

**Solutions:**

1. **Restart kernel and clear output:**
   - In Jupyter: Kernel → Restart & Clear Output

2. **Run cells one at a time** instead of "Run All"

3. **Reduce batch size / iterations:**
   ```python
   n_estimators=100  # Smaller value
   CV_FOLDS=3        # Fewer folds
   ```

4. **Close other applications** to free up memory

### Problem: Predictions All Same Class

**Cause:** Model severely overfit or data preprocessing issue.

**Solutions:**

1. **Check target distribution:**
   ```python
   print(df['Target'].value_counts())
   ```

2. **Verify preprocessing:**
   ```python
   print(f"Unique values in y: {np.unique(y)}")
   print(f"Class distribution: {np.bincount(y)}")
   ```

3. **Reduce overfitting:**
   ```python
   # Lower max_depth
   max_depth=3
   # Add regularization
   reg_alpha=1.0
   reg_lambda=1.0
   ```

4. **Check class weights:**
   ```python
   # Add to model initialization
   scale_pos_weight=class_weight_ratio
   ```

### Problem: Different Results Each Run

**Cause:** Random seed not properly set.

**Solution:**

1. **Verify random seed is set** in Section 1:
   ```python
   RANDOM_SEED = 42
   ```

2. **Check it's used in all functions:**
   - train_test_split
   - Model initialization
   - Cross-validation

3. **Set global seeds** (add to Section 1):
   ```python
   import random
   random.seed(RANDOM_SEED)
   np.random.seed(RANDOM_SEED)
   ```

---

## 📞 Additional Help

### Getting Support

1. **Check the Kaggle competition discussion forum** - Other competitors often share tips
2. **Review the competition rules** - Make sure your approach is allowed
3. **Read the evaluation metric** - Understand how your submission is scored
4. **Look at public kernels** - See what others are doing (but don't copy directly)

### Best Practices

- **Start simple, then iterate** - Don't enable everything at once
- **Track your experiments** - Keep notes on what works and what doesn't
- **Make frequent submissions** - Kaggle gives you multiple chances
- **Learn from your scores** - Compare local CV with leaderboard score

### Competition Strategy

1. **Quick baseline** (Day 1): Run notebook with defaults, make first submission
2. **Model comparison** (Day 2): Try all three models, pick the best
3. **Optimization** (Day 3-4): Hyperparameter tuning, feature engineering
4. **Ensembling** (Day 5): Combine your best models
5. **Final push** (Last day): Fine-tune and make final submissions

---

## 🎓 Learning Resources

### Understanding the Models

- **XGBoost**: [Official Documentation](https://xgboost.readthedocs.io/)
- **LightGBM**: [Official Documentation](https://lightgbm.readthedocs.io/)
- **CatBoost**: [Official Documentation](https://catboost.ai/docs/)

### Machine Learning Concepts

- **Cross-validation**: Why it's important and how it works
- **Overfitting**: How to detect and prevent it
- **Feature engineering**: Creating better features from raw data
- **Hyperparameter tuning**: Finding optimal model settings

### Kaggle-Specific

- **Kaggle Learn**: Free micro-courses on ML and competitions
- **Kaggle Notebooks**: Learn from top-performing solutions
- **Competition Forums**: Ask questions and share knowledge

---

## ✅ Pre-Submission Checklist

Before making your final submission, verify:

- [ ] All required files are in place (`data.csv`, `test.csv`)
- [ ] No errors when running the complete notebook
- [ ] Submission file is created successfully
- [ ] Submission file has correct format (id, Target columns)
- [ ] No missing values in submission
- [ ] Number of predictions matches test set size
- [ ] Prediction distribution looks reasonable
- [ ] CV score is documented for your records
- [ ] Model files are saved (in case you need to reproduce results)

---

## 📄 License and Credits

This notebook is provided as-is for the AIDAMS ML competition. Feel free to modify and adapt it to your needs.

**Good luck with the competition! 🚀**

---

*Last updated: 2026*
