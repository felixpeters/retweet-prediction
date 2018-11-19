import twitter
import numpy as np

def extract_retweet_counts(tweets):
    """
    Returns retweet counts suited for regression models.

    Args:
        tweets: Array of twitter.models.status instances
    Returns:
        Numpy array with retweet counts for the given tweets
    """
    labels = np.array([t.retweet_count or 0 for t in tweets])
    return labels
