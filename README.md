 # SC4002-Natural-Language-Processing

 ### 📘 Project Overview  
This project implements multiple deep learning architectures for **TREC Question Classification**, which involves classifying open-domain questions into six broad categories:  

**HUM (Human), ENTY (Entity), DESC (Description), NUM (Numerical), LOC (Location), ABBR (Abbreviation)**  

We explore:  
- Word embedding preparation using **GloVe (300d)**  
- OOV mitigation strategies without transformer models  
- RNN-based classification with various regularization methods  
- Enhanced architectures (BiLSTM, BiGRU, CNN, Ensemble)  
- Class imbalance handling using **weighted cross-entropy loss**

---

### 📂 Dataset Preparation  
- **Dataset:** [TREC Question Classification Dataset](https://cogcomp.seas.upenn.edu/Data/QA/QC/)  
- **Tokenization:** Performed using `spaCy (en_core_web_sm)`  
- **Normalization:** Lowercased tokens while retaining punctuation for syntactic cues  
- **Data Split:** 80% training / 20% validation split for tuning and early stopping  

---

### Part 1 — Word Embeddings  

**Embedding Source:** GloVe 6B (300 dimensions)  

**Padding:** `<pad>` initialized to zero vector  
**Trainability:** All embeddings trainable for domain adaptation  

#### OOV Strategy  
1. Attempt lowercase match in vocabulary  
2. Average available subword or partial matches  
3. Fall back to Gaussian random initialization if unseen  

#### Visualization  
- Top 20 frequent words per topic projected using **t-SNE**  
- Observed partial clustering:  
  - **LOC** words cluster tightly  
  - **DESC** and **ENTY** are more dispersed  

---

### Part 2 — Model Training & Evaluation (RNN)  

#### Model Configuration  
| Parameter | Value |
|------------|--------|
| Hidden Size | 256 |
| Layers | 2 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Dropout | 0.5 |
| L2 Weight Decay | 1e-4 |
| Epochs | 20 |
| Early Stopping | Patience = 10 |

**Best Model:** RNN + Dropout + L2 → **Test Accuracy: 0.9153**

#### Regularization Comparison  
| Strategy | Description | Test Accuracy |
|-----------|--------------|----------------|
| None | Baseline | 0.8899 |
| Dropout (0.5) | Random neuron disable | 0.8948 |
| L2 Regularization | Weight decay 1e-4 | 0.8992 |
| **Dropout + L2** | Combined regularization | **0.9153** |

#### Sentence Representation Strategies  
| Aggregation | Description | Test Accuracy |
|--------------|--------------|----------------|
| Last Hidden State | Concatenated forward/backward | **0.9142** |
| Mean Pooling | Average of all hidden states | 0.8875 |
| Max Pooling | Element-wise maximum | 0.9114 |

#### Topic-wise Accuracy  
| Topic | Accuracy |
|--------|-----------|
| ABBR | 0.7778 |
| ENTY | 0.8298 |
| HUM | 0.8519 |
| NUM | 0.8850 |
| LOC | 0.9077 |
| DESC | **0.9710** |

---

### Part 3 — Model Enhancement  

#### Model Comparisons  
| Model | Validation Accuracy | Test Accuracy |
|--------|----------------------|----------------|
| BiLSTM | 0.878 | 0.8894 |
| BiGRU | 0.881 | **0.9055** |
| CNN | 0.889 | 0.9050 |
| **Ensemble (BiGRU + CNN)** | **0.897** | **0.9114** |

#### Ensemble Strategy  
Combines **BiGRU** (captures long-range dependencies) with **CNN** (extracts strong local features).  
This hybrid approach leverages sequential context and localized phrase-level information for improved generalization.  

#### Class Imbalance Improvement  
- Implemented **class-weighted CrossEntropyLoss**  
- Penalizes misclassification in minority classes (e.g., ABBR, LOC)  
- Formula:  Wc = N / (K * nc)
where  
- *Wc* = class weight  
- *N* = total samples  
- *K* = number of classes  
- *nc* = number of samples per class  

---

### 🧠 Key Insights  
- Trainable GloVe embeddings achieved strong baseline performance (>0.91)  
- Dropout + L2 effectively reduced overfitting  
- **BiGRU** surpassed BiLSTM on short sequences due to simpler gating dynamics  
- **Ensemble models** improved overall robustness  
- **Weighted loss** mitigated imbalance but slightly reduced performance in some majority classes  

---


  
