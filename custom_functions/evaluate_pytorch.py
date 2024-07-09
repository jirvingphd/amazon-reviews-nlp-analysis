import dojo_ds as fn
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd

from collections import Counter
import numpy as np
import re
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# Function to tokenize and preprocess the text
def preprocess_text(text):
    """
    Preprocesses the given text by removing punctuation, converting it to lowercase,
    and splitting it into a list of words.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of words after preprocessing the input text.
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def get_vocab(X):
    """
    Create a vocabulary dictionary from a list of texts.

    Args:
        X (list): A list of texts.

    Returns:
        dict: A dictionary where the keys are unique words from the texts and the values are their corresponding indices.
    """
    all_words = [word for review in X for word in preprocess_text(review)]
    vocab = Counter(all_words)
    vocab = {word: i+1 for i, (word, _) in enumerate(vocab.items())}  # Start indexing from 1
    return vocab
# # Create a vocabulary from the training data
# all_words = [word for review in X_train for word in preprocess_text(review)]
# vocab = Counter(all_words)
# vocab = {word: i+1 for i, (word, _) in enumerate(vocab.items())}  # Start indexing from 1

# Tokenizer function
def tokenizer(text, X_train):
    """
    Tokenizes the given text using the vocabulary obtained from X_train.

    Args:
        text (str): The input text to be tokenized.
        X_train (list): The training data used to build the vocabulary.

    Returns:
        torch.Tensor: A tensor containing the tokenized representation of the input text.
    """
    vocab = get_vocab(X_train)
    tokens = preprocess_text(text)
    return torch.tensor([vocab.get(word, 0) for word in tokens], dtype=torch.long)  # 0 for unknown words

# Padding function to make all sequences the same length
def pad_collate_fn(batch):
    """
    Function to collate and pad a batch of data.

    Args:
        batch (list): A list of tuples, where each tuple contains two elements: xx and yy.

    Returns:
        tuple: A tuple containing the padded xx and yy tensors.
    """
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)  # Padding with 0
    yy = torch.tensor(yy, dtype=torch.float32)
    return xx_pad, yy


# Define a custom Dataset class
class AmazonReviewsDataset(Dataset):
    """
    A custom PyTorch dataset for Amazon reviews.

    Args:
        reviews (pandas.Series): The reviews data.
        labels (pandas.Series): The corresponding labels for the reviews.
        tokenizer (callable): A tokenizer function to tokenize the reviews.

    Attributes:
        reviews (pandas.Series): The reviews data.
        labels (pandas.Series): The corresponding labels for the reviews.
        tokenizer (callable): A tokenizer function to tokenize the reviews.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the tokenized review and its label at the given index.
        __repr__(): Returns a string representation of the dataset.

    """

    def __init__(self, reviews, labels, tokenizer):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews.iloc[idx]
        label = self.labels.iloc[idx]
        tokens = self.tokenizer(review)
        return tokens, label
    
    def __repr__(self):
        # Display the number of samples
        info = f"AmazonReviewsDataset with {len(self)} samples\n"
        # Display the first few samples
        info += "First few samples:\n"
        for i in range(min(3, len(self))):
            review = self.reviews.iloc[i]
            label = self.labels.iloc[i]
            info += f"{i+1}. Label: {label}; Review: {review}\n"
        return info
    

# class History():
#     def __init__(self, history_dict=None):
#         if history_dict is None:
#             history_dict = {
#                 'loss': [],
#                 'val_loss': [],
#                 'accuracy': [],
#                 'val_accuracy': []
#             }
#         self.history = history_dict
#         n_epochs = len(history_dict['loss'])
#         self.epoch = list(range(1, n_epochs+1))
        

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    """
    Trains a PyTorch model using the specified criterion, optimizer, and data loaders.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        criterion (torch.nn.Module): The loss function to optimize.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        epochs (int): The number of epochs to train the model.

    Returns:
        History: An instance of the History class containing the training history.

    """
    # Initialize the history dictionary
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [],
        'val_accuracy': []
    }
    
    class History():
        def __init__(self, history_dict=None):
            if history_dict is None:
                history_dict = {
                    'loss': [],
                    'val_loss': [],
                    'accuracy': [],
                    'val_accuracy': []
                }
            self.history = history_dict
            n_epochs = len(history_dict['loss'])
            self.epoch = list(range(1, n_epochs+1))
            
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        # Initialize the running loss and correct predictions counters
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Loop over the training data
        for inputs, labels in train_loader:
            labels = labels.unsqueeze(1).float()  # Reshape labels for compatibility with BCEWithLogitsLoss
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize
            running_loss += loss.item()  # Update running loss

            # Get predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        # Calculate the training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                labels = labels.unsqueeze(1)  # Reshape labels for compatibility with BCEWithLogitsLoss
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                val_running_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct_predictions += (preds == labels).sum().item()
                val_total_predictions += labels.size(0)

        # Calculate the validation loss and accuracy
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        msg = f'- Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}; '
        msg += f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}'
        print(msg)
    
    return History(history)
        
        
# Evaluation function
def get_predictions(model, dataloader, convert=True):
    """
    Get predictions from a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to use for prediction.
        dataloader (torch.utils.data.DataLoader): The data loader containing the input data.
        convert (bool, optional): Whether to convert the predictions to sklearn classes. 
            Defaults to True.

    Returns:
        tuple: A tuple containing the true labels and predicted labels.

    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in dataloader:
            labels = labels.unsqueeze(1)  # Reshape labels for compatibility with BCELoss
            outputs = model(inputs)  # Forward pass
            preds = (outputs > 0.5).float()  # Convert outputs to binary predictions
            all_labels.extend(labels.numpy())  # Collect true labels
            all_preds.extend(preds.numpy())  # Collect predictions
            
    if convert:
        all_labels = convert_y_to_sklearn_classes(all_labels)
        all_preds = convert_y_to_sklearn_classes(all_preds)
    return all_labels, all_preds


def convert_y_to_sklearn_classes(y, verbose=False):
    """
    Helper function to convert neural network outputs to class labels.

    Args:
        y (array/Series): Predictions to convert to classes.
        verbose (bool, optional): Print which preprocessing approach is used. Defaults to False.

    Returns:
        array: Target as 1D class labels
    """
    import numpy as np
    if isinstance(y,list):
        y = np.array(y)
    # If already one-dimension
    if np.ndim(y) == 1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y

    # If 2 dimensions with more than 1 column:
    elif y.shape[1] > 1:
        if verbose:
            print("- y is 2D with >1 column. Using argmax for metrics.")
        return np.argmax(y, axis=1)

    else:
        if verbose:
            print("y is 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)
    
    
    
def classification_metrics(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False, values_format=".2f",
                           target_names = None, return_fig=True):
    """
    Compute classification metrics and display confusion matrices.

    Parameters:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - label (str): Label for the classification metrics.
    - output_dict (bool): Whether to return the classification report as a dictionary.
    - figsize (tuple): Figure size for the confusion matrix plot.
    - normalize (str): Normalization method for the confusion matrix. Default is 'true'.
    - cmap (str): Colormap for the confusion matrix plot. Default is 'Blues'.
    - colorbar (bool): Whether to display a colorbar for the confusion matrix plot.
    - values_format (str): Format for the values in the confusion matrix. Default is ".2f".
    - target_names (list): List of target class names.
    - return_fig (bool): Whether to return the figure object.

    Returns:
    - If output_dict is True, returns a dictionary containing the classification report.
    - If return_fig is True, returns the figure object.

    """
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    import numpy as np
    # Get the classification report
    report = classification_report(y_true, y_pred,target_names=target_names)
    
    ## Print header and report
    header = "-"*70
    print(header, f" Classification Metrics: {label}", header, sep='\n')
    print(report)
    
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0], 
                                            display_labels=target_names);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            colorbar=colorbar,
                                            ax = axes[1],
                                            display_labels=target_names);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()
    plt.show()
    
    # Return dictionary of classification_report
    if output_dict==True:
        report_dict = classification_report(y_true, y_pred,target_names=target_names, output_dict=True)
        return report_dict

    elif return_fig == True:
        return fig
    
    
def plot_history(history, figsize=(6,8), return_fig=False):
    """
    Plots the training and validation metrics from the given history object.

    Parameters:
    - history: A history object containing the training and validation metrics.
    - figsize: Optional. A tuple specifying the size of the figure. Default is (6, 8).
    - return_fig: Optional. If True, the function will return the figure object instead of displaying it. Default is False.

    Returns:
    - If return_fig is True, the function returns the figure object.
    - If return_fig is False, the function displays the figure.

    Example usage:
    >>> plot_history(history, figsize=(8, 6), return_fig=True)
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Get a unique list of metrics 
    all_metrics = np.unique([k.replace('val_','') for k in history.history.keys()])

    # Plot each metric
    n_plots = len(all_metrics)
    fig, axes = plt.subplots(nrows=n_plots, figsize=figsize)
    axes = axes.flatten()

    # Loop through metric names add get an index for the axes
    for i, metric in enumerate(all_metrics):
        # Get the epochs and metric values
        epochs = history.epoch
        score = history.history[metric]

        # Plot the training results
        axes[i].plot(epochs, score, label=metric, marker='.')

        # Plot val results (if they exist)
        try:
            val_score = history.history[f"val_{metric}"]
            axes[i].plot(epochs, val_score, label=f"val_{metric}",marker='.')
        except:
            pass
        finally:
            axes[i].legend()
            axes[i].set(title=metric, xlabel="Epoch",ylabel=metric)
   
    # Adjust subplots and show
    fig.tight_layout()
 
    if return_fig:
        return fig
    else:
        plt.show()
    
def evaluate_classification_pytorch(model, 
                                    X_train=None, y_train=None, 
                                    X_test=None, y_test=None,
                                    history=None, history_figsize=(6,6),
                                    figsize=(6,4), normalize='true',
                                    output_dict=False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                    colorbar=False, target_names=None, 
                                    return_fig=False):
    """
    Evaluate the performance of a PyTorch classification model on training and test data.

    Parameters:
    - model: The PyTorch model to evaluate.
    - X_train: The training data features. Default is None.
    - y_train: The training data labels. Default is None.
    - X_test: The test data features. Default is None.
    - y_test: The test data labels. Default is None.
    - history: The training history of the model. Default is None.
    - history_figsize: The figure size for plotting the training history. Default is (6, 6).
    - figsize: The figure size for plotting the classification metrics. Default is (6, 4).
    - normalize: Whether to normalize the confusion matrix. Default is 'true'.
    - output_dict: Whether to return the classification metrics as a dictionary. Default is False.
    - cmap_train: The color map for the training data confusion matrix. Default is 'Blues'.
    - cmap_test: The color map for the test data confusion matrix. Default is 'Reds'.
    - values_format: The format for displaying the values in the confusion matrix. Default is '.2f'.
    - colorbar: Whether to display the color bar in the confusion matrix. Default is False.
    - target_names: The names of the target classes. Default is None.
    - return_fig: Whether to return the figure object for the classification metrics. Default is False.

    Returns:
    - results_dict: A dictionary containing the classification metrics for the training and test data.
    """
    
    if (X_train is None) & (X_test is None):
        raise Exception('\nEither X_train & y_train or X_test & y_test must be provided.')
 
    shared_kwargs = dict(output_dict=True, 
                      figsize=figsize,
                      colorbar=colorbar,
                      values_format=values_format, 
                      target_names=target_names,)
    # Plot history, if provided
    if history is not None:
        fn.evaluate.plot_history(history, figsize=history_figsize)
    ## Adding a Print Header
    print("\n"+'='*80)
    print('- Evaluating Network...')
    print('='*80)
    ## TRAINING DATA EVALUATION
    # check if X_train was provided
    if X_train is not None:
        ## Check if X_train is a dataset
        # Get predictions for training data
        y_train, y_train_pred = get_predictions(model, dataloader=X_train, convert=True)
        
        # Call the helper function to obtain metrics for training data
        results_train = classification_metrics(y_train, y_train_pred, cmap=cmap_train,label='Training Data', **shared_kwargs)
        
        # ## Run model.evaluate         
        # print("\n- Evaluating Training Data:")
        # print(model.evaluate(X_train, return_dict=True))
    
    # If no X_train, then save empty list for results_train
    else:
        results_train = None
  
  
    ## TEST DATA EVALUATION
    # check if X_test was provided
    if X_test is not None:
        y_test, y_test_pred = get_predictions(model, dataloader=X_test, convert=True)
        
        
        # Call the helper function to obtain metrics for test data
        results_test = classification_metrics(y_test, y_test_pred, cmap=cmap_test,label='Test Data', **shared_kwargs)
        
        ## Run model.evaluate         
        # print("\n- Evaluating Test Data:")
        # print(model.evaluate(X_test, return_dict=True))
      
    # If no X_test, then save empty list for results_test
    else:
        results_test = None
      
    if (output_dict == True) | (return_fig==True):
        # Store results in a dataframe if ouput_frame is True
        results_dict = {'train':results_train,
                        'test': results_test}
        return results_dict
