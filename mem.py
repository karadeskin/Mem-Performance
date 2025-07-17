#importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- data preprocessing ---
dataset = pd.read_csv('Memory_Performance.csv')

def run_eda(df):
    print("\n--- Dataset Head ---")
    print(df.head())
    print("\n--- Dataset Description ---")
    print(df.describe())
    print("\n--- Dataset Info ---")
    print(df.info())
    numeric_cols = ['TimeSinceStudy', 'Repetitions', 'StudyTime', 'RetentionScore']
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'plots/hist_{col}.png')  
        plt.show()
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')  
    plt.show()
    plt.figure()
    sns.countplot(x='Difficulty', data=df)
    plt.title('Count of Difficulty Levels')
    plt.tight_layout()
    plt.savefig('plots/count_difficulty.png')  
    plt.show()
    plt.figure()
    sns.boxplot(x='Difficulty', y='RetentionScore', data=df)
    plt.title('RetentionScore by Difficulty Level')
    plt.tight_layout()
    plt.savefig('plots/boxplot_retention_by_difficulty.png')  
    plt.show()

run_eda(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- building the ANN ---
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))
ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, batch_size=32, epochs=100)

# --- making predictions and evaluating the model ---
y_pred = ann.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# --- visualizing actual vs predicted ---
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Retention Score')
plt.ylabel('Predicted Retention Score')
plt.title('Actual vs Predicted Retention Scores')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()