#-------Importing Required Libraries-------#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from knn import KNN  # Importing the custom KNN implementation

# Load the data from the csv file
IRIS_DATA_PATH = r'C:\Users\HP\Documents\Machine Learning\KNN\dataset\iris.csv'


iris_df = pd.read_csv(IRIS_DATA_PATH)

# Convert categorical values to numerical values
iris_df['species'] = iris_df['species'].map({
    'Iris-setosa': 0, 
    'Iris-versicolor': 1, 
    'Iris-virginica': 2
})

# Split the data into features and target
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris_df['species'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1234, shuffle=True, stratify=y)

labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# Store accuracy scores for visualization
accuracy_scores = []

# Iterate through different values of k
for i in range(1, 10):
    knn = KNN(k=i)  # Initialize the KNN with k neighbors
    knn.fit(X_train, y_train)  # Train the model
    predictions = knn.predict(X_test)  # Predict the test set
    
    # Print accuracy, precision, recall, f1-score
    print(f"K = {i}")
    print(classification_report(y_test, predictions, target_names=labels))
    
    # Append accuracy score for plotting
    accuracy_scores.append(accuracy_score(y_test, predictions))


# Plot accuracy vs k
sns.lineplot(x=range(1, len(accuracy_scores) + 1), y=accuracy_scores)
plt.title('KNN Accuracy vs K (Neighbours)')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

# Determine the optimal k
optimal_k = accuracy_scores.index(max(accuracy_scores)) + 1
print(f"Optimal k: {optimal_k} with accuracy: {max(accuracy_scores):.2f}")
