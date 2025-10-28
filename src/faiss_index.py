import os
import faiss
import numpy as np
import itertools
from sklearn.metrics import pairwise_distances


class FaceIndex:
    """ Face Index using FAISS for efficient similarity search """

    def __init__(self, dim=128, index_path="face_index.bin"):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.labels = []
        self.embeddings = []  # lưu embedding để phân tích threshold sau này
        self.threshold = None  # có thể được thiết lập tự động

        if os.path.exists(index_path):
            self.loadIndex()

    def create(self, folder_path, embedding_extractor):
        """Create FAISS index from images in the specified folder."""

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                person_name = filename.split("_")[0]
                img_path = os.path.join(folder_path, filename)
                # Giả sử có hàm getEmbeddings để lấy embedding từ ảnh
                embeddings = embedding_extractor.getEmbeddings(img_path)  # Hàm này cần được định nghĩa ở nơi khác
                for emb in embeddings:
                    self.add(emb, person_name)

        self.saveIndex()

    def add(self, embedding, label):
        """Add a new face embedding to the index."""

        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        print("Index dim:", self.index.d)
        print("Embedding shape:", embedding.shape)

        self.index.add(embedding.astype(np.float32))
        self.labels.append(label)
        self.embeddings.append(embedding.astype(np.float32))

    def search(self, embedding, top_k=1, threshold=None):
        """Search for most similar embeddings with optional distance threshold."""

        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)

        distances, indices = self.index.search(embedding.astype(np.float32), top_k)
        result_labels = []

        # Chọn ngưỡng threshold từ đối số hoặc thuộc tính của class
        if threshold is None:
            threshold = self.threshold if self.threshold is not None else float('inf')

        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.labels) and dist <= threshold:
                result_labels.append(self.labels[idx])
            else:
                result_labels.append("Unknown")

        return result_labels, distances[0]

    def suggestOptimalThreshold(self):
        """Estimate optimal threshold based on intra/inter-class distances."""

        if len(self.embeddings) < 2:
            print("⚠️ Không đủ dữ liệu để tính threshold (cần ≥ 2 embeddings).")
            return None

        embeddings = np.vstack(self.embeddings)
        labels = np.array(self.labels)
        distances = pairwise_distances(embeddings, metric='euclidean')

        intra_dists = []
        inter_dists = []

        for i, j in itertools.combinations(range(len(labels)), 2):
            if labels[i] == labels[j]:
                intra_dists.append(distances[i, j])
            else:
                inter_dists.append(distances[i, j])

        if not intra_dists or not inter_dists:
            print("⚠️ Cần ít nhất 2 nhãn khác nhau để ước lượng threshold.")
            return None

        mean_intra = np.mean(intra_dists)
        mean_inter = np.mean(inter_dists)
        threshold = (mean_intra + mean_inter) / 2

        print(f"📊 Trung bình intra-class: {mean_intra:.4f}")
        print(f"📊 Trung bình inter-class: {mean_inter:.4f}")
        print(f"✅ Gợi ý threshold tối ưu: {threshold:.4f}")

        self.threshold = threshold
        return threshold

    def saveIndex(self):
        """Save the FAISS index and labels to disk."""

        faiss.write_index(self.index, self.index_path)
        np.save(self.index_path.replace(".bin", "_labels.npy"), np.array(self.labels))
        if self.embeddings:
            np.save(self.index_path.replace(".bin", "_embeddings.npy"), np.vstack(self.embeddings))
        if self.threshold is not None:
            with open(self.index_path.replace(".bin", "_threshold.txt"), "w") as f:
                f.write(str(self.threshold))

    def loadIndex(self):
        """Load the FAISS index, labels, embeddings, and threshold."""

        self.index = faiss.read_index(self.index_path)

        label_path = self.index_path.replace(".bin", "_labels.npy")
        if os.path.exists(label_path):
            self.labels = np.load(label_path, allow_pickle=True).tolist()

        emb_path = self.index_path.replace(".bin", "_embeddings.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path, allow_pickle=True).tolist()

        thr_path = self.index_path.replace(".bin", "_threshold.txt")
        if os.path.exists(thr_path):
            with open(thr_path, "r") as f:
                self.threshold = float(f.read().strip())