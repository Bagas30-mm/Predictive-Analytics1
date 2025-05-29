# %% [markdown]
# # Documentation
# 

# %% [markdown]
# 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('personality_dataset.csv')
df.head()

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# # Exploratory Data Analysis (EDA)
# 

# %% [markdown]
# ## Bivariate Analysis
# 

# %% [markdown]
# ### Time Spent Alone
# 

# %%
plt.figure(figsize=(12, 7))
for personality in df['Personality'].unique():
    sns.kdeplot(data=df[df['Personality'] == personality], x='Time_spent_Alone', label=personality, common_norm=False, fill=True, alpha=0.5)
plt.title('Distribution of Time Spent Alone by Personality Type', fontsize=14, pad=15)
plt.xlabel('Hours Spent Alone per Day', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Stage Fear
# 

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Stage_fear', hue='Personality')
plt.title('Distribution of Stage Fear by Personality Type', fontsize=14, pad=15)
plt.xlabel('Stage Fear', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Social event attendance
# 

# %%
plt.figure(figsize=(12, 7))
for personality in df['Personality'].unique():
    sns.kdeplot(data=df[df['Personality'] == personality], x='Social_event_attendance', label=personality, common_norm=False, fill=True, alpha=0.5)
plt.title('Distribution of Time Spent Alone by Personality Type', fontsize=14, pad=15)
plt.xlabel('Hours Spent Alone per Day', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Going outside
# 

# %%
plt.figure(figsize=(12, 7))
for personality in df['Personality'].unique():
    sns.kdeplot(data=df[df['Personality'] == personality], x='Going_outside', label=personality, common_norm=False, fill=True, alpha=0.5)
plt.title('Distribution of Time Spent Alone by Personality Type', fontsize=14, pad=15)
plt.xlabel('Hours Spent Alone per Day', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Drained_after_socializing
# 

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Drained_after_socializing', hue='Personality')
plt.title('Distribution of Stage Fear by Personality Type', fontsize=14, pad=15)
plt.xlabel('Stage Fear', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Friends circle size
# 

# %%
plt.figure(figsize=(12, 7))
for personality in df['Personality'].unique():
    sns.kdeplot(data=df[df['Personality'] == personality], x='Friends_circle_size', label=personality, common_norm=False, fill=True, alpha=0.5)
plt.title('Distribution of Time Spent Alone by Personality Type', fontsize=14, pad=15)
plt.xlabel('Hours Spent Alone per Day', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Post frequency
# 

# %%
plt.figure(figsize=(12, 7))
for personality in df['Personality'].unique():
    sns.kdeplot(data=df[df['Personality'] == personality], x='Post_frequency', label=personality, common_norm=False, fill=True, alpha=0.5)
plt.title('Distribution of Time Spent Alone by Personality Type', fontsize=14, pad=15)
plt.xlabel('Hours Spent Alone per Day', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Personality Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Multivariate Analysis
# 

# %%
# Create a copy of the dataframe to avoid modifying the original
df_encoded = df.copy()

# Encode categorical variables
categorical_columns = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for column in categorical_columns:
    df_encoded[column] = df_encoded[column].map({'Yes': 1, 'No': 0} if column != 'Personality' else {'Introvert': 0, 'Extrovert': 1})

# Handle missing values
df_encoded = df_encoded.dropna()

# %%
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# %% [markdown]
# # Preprocessing
# 

# %%
df_encoded = df_encoded.drop(columns=['Stage_fear', 'Drained_after_socializing'])

# %%
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# %% [markdown]
# # Modelling
# 

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare features (X) and target (y)
X = df_encoded.drop('Personality', axis=1)
y = df_encoded['Personality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Test Lazy Predict to get the best model
# 

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lazypredict.Supervised import LazyClassifier

# Prepare features (X) and target (y)
X = df_encoded.drop('Personality', axis=1)
y = df_encoded['Personality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit LazyClassifier
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)

# Print the models' performance summary
models

# %% [markdown]
# ## SVC
# 

# %%
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

# Initialize the SVC model
model = SVC(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# %%
# Create confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Introvert', 'Extrovert'], 
            yticklabels=['Introvert', 'Extrovert'])
plt.title('Confusion Matrix', fontsize=14, pad=15)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.show()

# %%
from sklearn.inspection import permutation_importance

# Create feature importance plot using permutation importance

# Calculate permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Create feature importance plot
plt.figure(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance for Personality Prediction', fontsize=14, pad=15)
plt.xlabel('Mean Permutation Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Usage
# 

# %%
def predict_personality(time_spent_alone, social_event_attendance, going_outside, friends_circle_size, post_frequency):
    """
    Predicts the personality type (Introvert or Extrovert) based on the given features.

    Parameters:
    - time_spent_alone (float): Time spent alone per day.
    - social_event_attendance (float): Frequency of social event attendance.
    - going_outside (float): Frequency of going outside.
    - friends_circle_size (float): Size of friends circle.
    - post_frequency (float): Frequency of posting on social media.
    - stage_fear (str): Whether the person has stage fear ('Yes' or 'No').
    - drained_after_socializing (str): Whether the person feels drained after socializing ('Yes' or 'No').

    Returns:
    - str: Predicted personality type ('Introvert' or 'Extrovert').
    """

    # Create a DataFrame from the input features
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_spent_alone],
        'Social_event_attendance': [social_event_attendance],
        'Going_outside': [going_outside],
        'Friends_circle_size': [friends_circle_size],
        'Post_frequency': [post_frequency]
    })

    # Select only the features used for training
    input_data = input_data[X_train.columns]

    # Make the prediction
    prediction = model.predict(input_data)

    # Convert the prediction to personality type
    personality_type = 'Extrovert' if prediction[0] == 1 else 'Introvert'

    return personality_type

# Example usage:
time_spent_alone = 9
social_event_attendance = 0
going_outside = 0
friends_circle_size = 0
post_frequency = 3

predicted_personality = predict_personality(time_spent_alone, social_event_attendance, going_outside, friends_circle_size, post_frequency)
print(f"Predicted personality: {predicted_personality}")



