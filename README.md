MicroGPT Domain Name Generator 🚀
An AI-powered, end-to-end pipeline for generating, verifying, and scoring premium, brandable domain names using a lightweight Transformer architecture.

📖 Overview
This project utilizes a custom-trained character-level language model to generate highly brandable and phonetically appealing domain names. Instead of randomly combining letters, the neural network learns the latent structures, morphemes, and syllables of the most successful websites on the internet. It then generates novel, unregistered domains and evaluates them based on commercial value, length, and pronounceability.

🙏 Acknowledgments & Credits
This project stands on the shoulders of giants. I would like to explicitly acknowledge:

Andrej Karpathy: The core neural network architecture is based on his phenomenal microgpt (and makemore/nanoGPT) educational repositories.

GitHub Community Optimizations: I utilized a high-speed NumPy backend modification proposed by the GitHub community, which significantly accelerates local CPU training and inference without the overhead of heavy deep learning frameworks.

📊 Dataset
The model was trained on the Tranco List, a research-oriented top 1 Million websites dataset.
Before training, the dataset underwent a rigorous cleaning process (removing subdomains, TLDs, invalid characters, and noise) to ensure the model focuses purely on the core brandable strings. The final cleaned dataset provided a rich vocabulary of successful digital brand names.

✨ Key Features
Optimized Local Training: Uses a custom block_size and a fast NumPy backend to train a small Transformer model directly on your local machine, caching the weights (.pkl) for future use.

Smart Generation: Generates novel strings by sampling from the learned probability distributions (with adjustable Temperature for creativity).

Concurrent DNS Checker: Includes a safe, multi-threaded DNS resolution script that checks domain availability in bulk. It implements smart batching and delays to prevent ISP blocking or modem overload.

Premium Domain Scorer (Heuristic Evaluation): A post-processing script that scores available domains based on:

Length: Rewards shorter domains (5-6 characters).

Pronounceability: Evaluates the vowel-to-consonant ratio and penalizes unpronounceable consonant clusters.

Commercial Keywords: Boosts scores for domains containing valuable tech and business keywords (e.g., ai, tech, app, host, pay).

⚙️ Project Structure & Workflow
dataset_prep.py: Cleans and tokenizes the raw Tranco Top 1M list.

benchmark.py / train.py: Trains the microGPT model using the high-speed NumPy backend and saves the weights to microgpt_model_cache.pkl.

generate_domains.py: Loads the trained model, generates unique domain names, and concurrently checks their .com availability via DNS lookups, saving the open ones to available_domains_com.txt.

score_domains.py: Reads the available domains and applies heuristic scoring, outputting a ranked list of premium domains to premium_domains_com.txt.

🚀 Results
In a standard benchmark run:

Training: Achieved stable loss convergence (~2.2 - 2.3) demonstrating an optimal balance between learning and avoiding overfitting.

Availability: Generated 1000 novel .com domains, with an impressive ~60% availability rate.

Quality: Successfully discovered highly valuable, short (4-5 letter) and keyword-rich domains that are ready for registration and branding.

🛠️ Usage
(You can add your specific installation instructions and command-line usage here, for example:)