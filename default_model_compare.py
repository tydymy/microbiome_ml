import numpy as np
import pandas as pd
import tensorflow as tf
import biom
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Clear the default graph
tf.compat.v1.reset_default_graph()

# Clear the TensorFlow session
tf.keras.backend.clear_session()

# Focal Loss implementation
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()) # Avoid division by zero
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = - alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
    return tf.reduce_mean(loss)

df = biom.load_table('data/feature-table.biom').to_dataframe(dense=True).T
metadata = pd.read_csv('data/filtered_metadata.tsv', sep='\t', index_col=0)
df['label'] = metadata.loc[metadata.index.isin(df.index)].sc2_status
# Split the data into features and labels
features = df.drop('label', axis=1)
labels = df['label'].map(lambda x: 0 if x =='negative' else 1)
num_features = features.shape[1]

# Convert the features and labels to NumPy arrays
features = features.to_numpy().astype(np.float32)
labels = labels.to_numpy().astype(np.float32)

# Normalize the features--> replace with rclr....
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
model.compile(optimizer='adam', loss=focal_loss, metrics=['AUC'])

from sklearn.model_selection import train_test_split

# Split the data into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Train the model
model.fit(features_train, labels_train, epochs=400, batch_size=32, validation_data=(features_test, labels_test))

model_rf = RandomForestClassifier()
model_lr = LogisticRegression()
model_svm = SVC(probability=True)

# List of models
models = [model_svm, model, model_rf, model_lr]
model_names = ['Support Vector Machine', 'Neural Network', 'Random Forest', 'Logistic Regression']

# Initialize lists for storing AUC scores
auc_scores = []

# Plot recall-precision curves for each model
plt.figure(figsize=(8, 6))

for model, name in zip(models, model_names):
    if name == 'Neural Network':
        # For the neural network model
        # Make predictions
        predictions = model.predict(features_test)
    else:
        # For other models
        # Train the model
        model.fit(features_train, labels_train)
        # Make predictions (probabilities)
        predictions = model.predict_proba(features_test)[:, 1]

    # Calculate precision and recall values
    precision, recall, _ = precision_recall_curve(labels_test, predictions)

    # Calculate AUC
    auc_score = auc(recall, precision)
    auc_scores.append(auc_score)

    # Plot the recall-precision curve
    plt.plot(recall, precision, label='{} (AUC = {:.3f})'.format(name, auc_score))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig('data/pr_curve_comparison.png')
plt.show()

# Print AUC scores
for name, auc_score in zip(model_names, auc_scores):
    print('{}: AUC = {:.3f}'.format(name, auc_score))
