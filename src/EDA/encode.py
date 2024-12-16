import os
import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import random

class ExploratoryAnalysis:
    
    def __init__(self, folder_path, file_to_class_mapping, is_random=True, n_samples=None):
        self.folder_path = folder_path
        self.file_to_class_mapping = file_to_class_mapping
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.all_comments = []
        self.all_classes = []
        self.num_samples = n_samples
        self.is_random = is_random

    def load_data(self):
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if file_name.endswith('.csv'):
                data = pd.read_csv(file_path)
                
                if 'comment' not in data.columns:
                    raise ValueError(f"The file {file_name} does not contain a 'comment' column.")
                
                class_label = self.file_to_class_mapping[file_name]
                
                self.all_comments.extend(data['comment'].astype(str).tolist())
                self.all_classes.extend([class_label] * len(data))

    def sample_data(self):
        N = self.num_samples

        sampled_comments = []
        sampled_classes = []
        unique_classes = set(self.all_classes)

        for class_label in unique_classes:
            indices = [i for i, label in enumerate(self.all_classes) if label == class_label]
            if len(indices) < N:
                raise ValueError(f"Not enough data in class '{class_label}' to sample {N} rows.")
            
            if self.is_random:
                sampled_indices = random.sample(indices, N)
            else:
                sampled_indices = indices[:N]
            
            sampled_comments.extend([self.all_comments[i] for i in sampled_indices])
            sampled_classes.extend([self.all_classes[i] for i in sampled_indices])
        
        return sampled_comments, sampled_classes

    def convert_sentences(self, comments):
        """
        Convert comments to embeddings using the SentenceTransformer model.
        """
        embeddings = self.model.encode(comments, show_progress_bar=True)
        return embeddings

    def plot_embeddings(self, embeddings, sampled_classes, method='umap'):
        """
        Plot a visualization of the embeddings using t-SNE or UMAP, with centroids for each class.
        """
        if method not in ['tsne', 'umap']:
            raise ValueError("Method must be 'tsne' or 'umap'.")

        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            title = "t-SNE Plot of Sentence Embeddings (Color-coded by Class)"
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
            title = "UMAP Plot of Sentence Embeddings (Color-coded by Class)"
        
        X_embedded = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        unique_classes = list(set(sampled_classes))

        for class_label in unique_classes:
            indices = [i for i, label in enumerate(sampled_classes) if label == class_label]
            class_points = X_embedded[indices]
            
            plt.scatter(class_points[:, 0], class_points[:, 1], label=class_label, alpha=0.7)
        
        plt.title(title)
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.legend()
        plt.show()

    def run(self, method):
        """
        Run the entire pipeline: sample data, convert sentences, and plot embeddings.
        """
        if not self.all_comments:
            self.load_data()
        
        sampled_comments, sampled_classes = self.sample_data()

        embeddings = self.convert_sentences(sampled_comments)
        
        self.plot_embeddings(embeddings, sampled_classes, method=method)
