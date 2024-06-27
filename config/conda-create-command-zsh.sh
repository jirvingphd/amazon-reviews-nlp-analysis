#!/bin/zsh

# Usage:
# 1. Make executable: chmod +x conda-create-command-zsh.sh
# 2. Run: ./conda-create-command-zsh.sh
# Create a new conda environment named 'nlp-env' with Python 3.10
conda create -n nlp-env python=3.10 -y

# Initialize conda for zsh (only needed if not already done)
conda init zsh

# Reload the shell to apply changes from `conda init`
source ~/.zshrc

# Activate the new environment
conda activate nlp-env

# Pip install custom package
pip install dojo-ds

# Install the required packages
conda install -y pandas numpy matplotlib seaborn nltk gensim pyldavis jupyter scikit-learn

# Install the kernel in jupyter
python -m ipykernel install --user --name=nlp-env --display-name="Python(nlp-env)"

# Additional installations if required
# conda install -y <other-packages>
# pip install <other-pip-packages>

conda install -c conda-forge spacy
python -m spacy download en_core_web_sm

pip install scipy==1.10.1
pip install missingno seaborn tabulate
pip install pydantic langchain_openai langchain_core
pip install tensorflow-macos
pip install tensorflow-metal
pip install -U protobuf
pip install transformers
# pip install tf-keras

# Deactivate the environment
conda deactivate

echo "Environment 'nlp-env' created and packages installed successfully."