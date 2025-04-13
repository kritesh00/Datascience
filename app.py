import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_curve, auc, precision_score, recall_score

# Streamlit app title
st.title("Customer Segmentation using K-Means Clustering")
st.write("Upload a customer dataset and explore customer segments interactively.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    customer_data = pd.read_csv(uploaded_file)
    st.write("### Preview of the Data:")
    st.dataframe(customer_data.head())
    
    # Selecting relevant features
    if 'Annual Income (k$)' in customer_data.columns and 'Spending Score (1-100)' in customer_data.columns:
        X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']].values
        
        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choosing the number of clusters dynamically
        st.sidebar.header("K-Means Clustering Settings")
        max_clusters = 10
        wcss = []
        
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
        
        # Plot the Elbow Method
        st.write("### The Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        ax.set_title("Elbow Method for Optimal K")
        st.pyplot(fig)
        
        # Select the number of clusters
        k = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)
        
        # Train the K-Means Model
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to the dataset
        customer_data['Cluster'] = clusters
        
        # Show clustered data
        st.write("### Clustered Data Preview:")
        st.dataframe(customer_data.head())
        
        # Plot the clusters
        st.write("### Customer Segmentation Visualization")
        fig, ax = plt.subplots()
        
        colors = ['green', 'red', 'yellow', 'blue', 'purple', 'cyan', 'orange', 'pink', 'brown', 'gray']
        
        for i in range(k):
            ax.scatter(X_scaled[clusters == i, 0], X_scaled[clusters == i, 1],
                       s=50, c=colors[i], label=f'Cluster {i+1}')
        
        # Plot centroids
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=200, c='black', marker='X', label='Centroids')
        
        ax.set_xlabel("Annual Income (Standardized)")
        ax.set_ylabel("Spending Score (Standardized)")
        ax.set_title("Customer Segmentation Clusters")
        ax.legend()
        st.pyplot(fig)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, clusters)
        st.write(f"### Silhouette Score: {silhouette_avg:.2f}")
        
        # Plot histogram and box plot for data distribution
        st.write("### Data Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.histplot(customer_data['Annual Income (k$)'], bins=20, kde=True, ax=ax[0])
        ax[0].set_title("Annual Income Distribution")
        
        sns.boxplot(x=customer_data['Spending Score (1-100)'], ax=ax[1])
        ax[1].set_title("Spending Score Distribution")
        
        st.pyplot(fig)
        
    else:
        st.error("The uploaded CSV file does not contain the required columns: 'Annual Income (k$)' and 'Spending Score (1-100)'.")
