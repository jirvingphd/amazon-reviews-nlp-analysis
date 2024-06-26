#!/bin/bash

# Create a new conda environment named 'nlp_project' with Python 3.8
conda create -n nlp-env python=3.10 -y

# Activate the new environment
source activate nlp-env


# Pip install custom package
pip install dojo-ds

# Install the required packages
conda install -y pandas numpy matplotlib seaborn nltk gensim pyldavis jupyter scikit-learn

# Install the kenrel in jupyter
python -m ipykernel install --user --name=nlp-env --display-name="Python(nlp-env)"

# Additional installations if required
# conda install -y <other-packages>
# pip install <other-pip-packages>

# Deactivate the environment
conda deactivate

echo "Environment 'nlp_project' created and packages installed successfully."