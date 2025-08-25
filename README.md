# LOX-CDR
Source code for LOX-CDR: Towards Logically Explainable Cross-Domain Recommendation


# LOX-CDR: Towards Logically Explainable Cross-Domain Recommendation

![Figure 2](https://github.com/bjtu-lucas-nlp/LOX-CDR/tree/main/figures/Fig-2.png)


## 📖 Overview

LOX-CDR is a novel framework for cross-domain recommendation that leverages logical reasoning and large language models to provide explainable recommendations across different domains. This work addresses the cold-start problem in recommender systems by transferring knowledge between domains while maintaining interpretability through logical explanations.

## 🌟 Key Features

- **Logical Explainability**: Provides transparent reasoning for cross-domain recommendations using logic-based approaches
- **LLM Integration**: Leverages the power of large language models for enhanced understanding of user preferences across domains
- **Flexible Architecture**: Supports multiple domain pairs and different recommendation scenarios
- **State-of-the-art Performance**: Outperforms existing CDR baselines across various metrics

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Install required packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/bjtu-lucas-nlp/LOX-CDR.git
cd LOX-CDR

# Install dependencies
pip install -e .
```

## 📊 Datasets

Our experiments use the Amazon Review Dataset with the following domain pairs:

| Domain Pair | Source → Target | Description |
|------------|-----------------|-------------|
| Pair 1 | Books → Movies | Related domains with similar user interests, Book->Movie |
| Pair 2 | Books → CDs | Related domains with similar user interests, Book->CDs |
| Pair 3 | Movies → CDs | Related domains with similar user interests, Movie->CDs |

### Data Preparation

```bash
# Download and preprocessed datasets

```

## 🏗️ Architecture

LOX-CDR consists of two main components:

1. **Cross-domain sentiment-aware paradigm**: a cross-domain sentiment-aware paradigm to link aspects from user review feedback and associate preferences across domains
2. **A logically-interpretable generator**: explicitly captures cross-domain reasoning paths.

## 🔬 Experiments

### Running Experiments

```bash

# With custom parameters
nohup python3 lox_cdr_main.py \
    --model_name=Mixtral2-ncf-prefix-GAN-Tune \
    --factor_num=128 \
    --ncf_layer_num=3 \
    --base_model=../../Mistral-7B-Instruct-v0.2 \
    --batch_size=48 \
    --num_epochs=4 \
    --data_path=Data/Amazon/Bk2CD \
    --output_dir=Model/book2cd/model \
    --training_output=Model/book2cd/sft \
    > training_log_Bk2CD.txt 2>&1 &

```

### Evaluation Metrics

- **Rating Prediction**: MAE, RMSE
- **Explainability**: Logic Coverage, Explanation Quality Score


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by BJTU LUCAS NLP Lab
</p>
