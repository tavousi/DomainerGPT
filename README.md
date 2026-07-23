# 🚀 DomainerGPT: Ultra-Lightweight Domain Name Generator

DomainerGPT is a specialized, ultra-lightweight generative language model built entirely on a custom `NumPy` backend. It is trained specifically to discover, predict, and generate highly brandable, premium domain names (e.g., 4-to-6 letter `.com` domains) by learning the phonetic patterns of the world's top websites.

This project is built for **Domain Flipping** and **Startup Naming**, providing an end-to-end pipeline from data extraction to automated DNS availability checking and domain scoring.

---

## 💰 The Business Value (Domain Flipping)
In the digital real estate market, short, pronounceable `.com` domains are highly liquid assets. Investors (Domainers) constantly hunt for 4-5 letter combinations that sound like modern tech startups. 

Instead of randomly guessing letters, **DomainerGPT** was trained on the **Tranco Top 1 Million** dataset. It learned the underlying "DNA" and syllable structures of successful websites. The model hallucinates *new*, statistically probable domain names, checks their availability in real-time, and scores them based on length and commercial viability.

### 🌟 Examples of Generated Premium Domains
Here are a few examples of the style of domains this AI generates (which could potentially be registered for standard fees and flipped for much higher margins):
*   `aexio.com` 
*   `quantex.com` 
*   `zenvo.com` 
*   `voyra.com` 
*   `nural.com` 

---

## 🧠 Architecture & Technical Details
- **Base Architecture:** Transformer Decoder (MicroGPT style).
- **Parameters:** ~101,000 (Extremely lightweight).
- **Vocabulary Size:** 27 tokens (a-z + special characters).
- **Backend:** 100% `NumPy` (`numpy_backend.py`). All PyTorch dependencies were stripped out to create an insanely fast, CPU-friendly training environment using analytical gradient computation.
- **Context Window (Block Size):** 24 characters (Optimized for domain lengths).

---

## 📂 Repository Structure

### 1. Model & Engine (Root)
*   `backends/`: Contains `numpy_backend.py`, the mathematical heart of the model handling vectorized matrix multiplications and analytical gradients without heavy ML frameworks.
*   `microGPT_SITES.py`: The main script for defining and orchestrating the model.
*   `data.py`: Handles tokenization and batching of the domain dataset.
*   `verify_gradients.py`: Utility to ensure the custom NumPy math is calculating gradients correctly.
*   `inspect_model.py`: A tool to parse and view the `microgpt_model_cache.pkl` weights and configuration.

### 2. Data Preparation (`DATA_prep/`)
A robust pipeline to clean and filter the raw web data:
*   `microgpt_prep.py` & `cleanTOclean.py`: Scripts to sanitize data, remove subdomains, and filter for `a-z` characters only.
*   `testDNS.py` & `microgpt_sites.py`: Utilities for checking domain validity and formatting lists.

### 3. Generation & Business Tools
*   `generate_domains.py`: Loads the trained model to generate new names and concurrently checks their real-time DNS availability.
*   `score_domains.py`: Post-processes the available domains to rank them based on length, pronounceability, and market value.

---

## 🛠️ Quick Start

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt

---

A. Prepare Data: Run the scripts in the DATA_prep folder to generate input.txt.

B. Train the Model: Run microGPT_SITES.py.

C. Generate & Check: generate_domains.py

D: Score Premium Domains: score_domains.py

---
   🤝 Credits & AcknowledgementsMy work on this project is fundamentally based on two incredible repositories, which I have heavily utilized and modified to create this specialized domain-generation pipeline:Andrej Karpathy's microgpt: The core algorithmic inspiration comes from Andrej Karpathy's original microgpt, which trains and runs a GPT in 243 lines of pure, dependency-free Python. You can find his foundational code here: karpathy/microgpt gist.  chanjoongx's microgpt-efficiency: To optimize the training process without relying on heavy frameworks like PyTorch, I heavily relied upon the microgpt-efficiency repository by chanjoongx.  This project explores the cost of computational efficiency by implementing the algorithm across different backends.  I specifically adapted their numpy backend, which replaces the scalar computation graph with vectorized matrix operations and hand-derived manual backward passes.  According to their benchmarks, utilizing this NumPy backend achieves a speedup of roughly 250× over the standard scalar implementation.  You can find their repository and benchmark details here: chanjoongx/microgpt-efficiency.
