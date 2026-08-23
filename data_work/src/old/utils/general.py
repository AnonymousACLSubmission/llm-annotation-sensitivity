import tiktoken
import numpy as np
import pandas as pd

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Return the number of tokens in `text` for the specified OpenAI model.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Define a helper function to compute the mode.
def mode_value(x):
    m = x.mode()
    # Return the first mode if available; otherwise, NaN.
    return m.iloc[0] if not m.empty else np.nan
