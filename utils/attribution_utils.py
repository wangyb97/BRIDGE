# %%
# import tensorflow as tf
import igrads
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
# %%
# def attribution(inputs, model, atype='IG', steps=50):
#     """Compute sequence attribution(s) for a given model and inputs.

#     Args:
#         inputs (tf.Tensor or np.ndarray): 2D (input_length, 4) tensor of onehot-encoded sequence.
#         model (Keras Model): Model to compute attribution for.

#     Returns:
#         tf.Tensor: Feature attributions.
#     """
    
#     # pred = model.predict(tf.expand_dims(inputs, axis=0))
#     inputs = inputs.unsqueeze(0)
#     pred = model(inputs)

#     if atype == 'IG':
#         return igrads.integrated_gradients(inputs, model, target_mask=pred, steps=steps)
#     elif atype == 'grad_x_input':
#         return igrads.grad_x_input(inputs, model, target_mask=pred)
#     else:
#         raise ValueError(f'Unrecognized attribution type {atype}.')
def attribution(inputs, structure, model, atype='IG', steps=50):
    """Compute sequence and structure attributions for a given model and inputs.

    Args:
        inputs (tf.Tensor or np.ndarray): 2D (input_length, 4) tensor of onehot-encoded sequence.
        structure (tf.Tensor or np.ndarray): 2D tensor representing the structure of the sequence.
        model (Keras Model): Model to compute attribution for.

    Returns:
        tf.Tensor: Feature attributions for both sequence and structure.
    """
    
    # Combine sequence and structure inputs (assuming structure is one-hot encoded or of suitable shape)
    combined_inputs = (inputs, structure)  # This could be adjusted based on model's input format
    
    # Add batch dimension
    combined_inputs = (inputs.unsqueeze(0), structure.unsqueeze(0))
    
    # Make predictions
    pred = model(combined_inputs)

    if atype == 'IG':
        # Compute integrated gradients for both sequence and structure inputs
        return igrads.integrated_gradients(combined_inputs, model, target_mask=pred, steps=steps)
    elif atype == 'grad_x_input':
        # Compute grad_x_input for both sequence and structure inputs
        return igrads.grad_x_input(combined_inputs, model, target_mask=pred)
    else:
        raise ValueError(f'Unrecognized attribution type {atype}.')


custom_color_scheme = {
    'A': '#268a34',  # Green
    'C': '#2c51aa',  # Blue
    'G': '#f8981c',  # Orange
    'U': '#ea2529'   # Red
}

def make_attribution_figure(a, ax):
    df = pd.DataFrame(a, columns=['A', 'C', 'G', 'U'])
    # logomaker.Logo(df, shade_below=.5, fade_below=.5, font_name='Arial Rounded MT Bold', ax=ax)
    logomaker.Logo(df, shade_below=.5, fade_below=.5, ax=ax, color_scheme=custom_color_scheme)
    ax.spines['bottom'].set_visible(False)
    ax.axhline(0, color='#ea2529', linewidth=1.5)
    
    
def visualize_track_attribution(track, attribution, sequence=None, title=None):
    nplots = 3 if sequence is not None else 2
    hratio = [5, 2, 0.3] if sequence is not None else [5, 2]
    
    fig, axs = plt.subplots(nplots, 1, figsize=(22, 5), gridspec_kw={'height_ratios': hratio})
    axs[0].set_title(title)
    axs[0].plot(track, color='red', label='Pred. Signal', linewidth=2)
    
    # if isinstance(attribution, tf.Tensor):
    #     attribution = attribution.numpy()
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()

    make_attribution_figure(attribution, axs[1])
    
    if sequence is not None:
        make_attribution_figure(sequence2onehot(sequence).numpy(), axs[2])
    
    for ax in axs:
        # remove x-axis margins
        ax.margins(x=0.005)
        
    
    # remove plot boarder (except for x-axis)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].get_xaxis().set_visible(False)
    
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)
    axs[1].get_xaxis().set_visible(False)

    if sequence is not None:
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['left'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        #axs[2].get_xaxis().set_visible(False)
        axs[2].get_yaxis().set_visible(False)
    
    return fig

def visualize_attribution_only(attribution):
    fig, ax = plt.subplots(1, 1, figsize=(22, 1.5))
    
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()
    
    make_attribution_figure(attribution, ax)
    
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return fig


import torch.nn.functional as F

def _to_probs(value, key):
    if '_profile' in key:
        value = F.softmax(value, dim=1)
    elif '_mixing_coefficient' in key:
        value = torch.sigmoid(value)
    else:
        raise ValueError(f'Unknown key: {key}')
    return value


# %%
def predict(inputs, model, to_probs=True):
    pred = model(inputs)
    if to_probs:
        pred = {key: _to_probs(value, key) for key, value in pred.items()}
    return pred

base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# %%
def sequence2int(sequence):
    return [base2int.get(base, 999) for base in sequence]

import torch

def sequences2inputs(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]
    return F.one_hot(torch.tensor([sequence2int(s) for s in sequences]), num_classes=4).float()


def sequence2onehot(sequence):
    return F.one_hot(torch.tensor(sequence2int(sequence)), num_classes=4).float()


# %%
def predict_from_sequence(sequences, model, **kwargs):
    one_hot = sequences2inputs(sequences)
    pred = predict(one_hot, model, **kwargs)
    
    if isinstance(sequences, str):
        pred = {key: value.squeeze(0) for key, value in pred.items()}
    
    return pred


# %%
import torch

def __predict(self, inputs, **kwargs):
    """Returns model predictions on inputs with logits to probs."""

    return predict(inputs, model=self, **kwargs)

def __predict_from_sequence(self, sequences, **kwargs):
    """Predicts on RNA/DNA sequences.

    Args:
        sequences (str, list): RNA/DNA sequence(s). If a string, it is assumed to be a single sequence.

    Returns:
        dict: Dictionary of predictions.
    """
    # Assume a preprocessing function is required for sequence input
    return predict_from_sequence(sequences, model=self, **kwargs)

def __explain(self, inputs, **kwargs):
    """Generate attributions or explanations for model predictions."""
    return attribution(inputs, self, **kwargs)

def __add_attributes_and_bound_methods(model):
    # Bind custom methods to the PyTorch model instance
    model.predict = __predict.__get__(model)
    model.predict_from_sequence = __predict_from_sequence.__get__(model)
    model.explain = __explain.__get__(model)
    

def load_model(model, filepath, **kwargs):
    
    model.load_state_dict(torch.load(filepath))
    __add_attributes_and_bound_methods(model)
    model.eval()
    return model
