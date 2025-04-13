# README.md

## Customer Segmentation using K-Means Clustering

This project is an interactive web application for customer segmentation using K-Means clustering. Built with Streamlit, the app allows users to upload customer datasets and visualize clusters dynamically.

### Features

- Upload and preview customer datasets.
- Perform K-Means clustering with adjustable number of clusters.
- Visualize the Elbow Method to determine the optimal number of clusters.
- Display cluster segmentation and visualization.
- Compute silhouette score for clustering performance.
- Display data distribution using histograms and box plots.

### Prerequisites

- Python 3.7+

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/kritesh00/Datascience.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository-name>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Folder Structure

- `app.py`: Main application file.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Files and folders to ignore in the repository.

### Usage

1. Upload a CSV file containing customer data. Ensure the dataset includes the columns `Annual Income (k$)` and `Spending Score (1-100)`.
2. Adjust the number of clusters using the sidebar slider.
3. Explore the segmentation results and visualizations interactively.

### Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

### License

This project is licensed under the MIT License.


### requirements.txt

streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn

