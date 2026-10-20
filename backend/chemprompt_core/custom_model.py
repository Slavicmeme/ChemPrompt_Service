# config/custom_model.py

def get_model_repo(model_repo):
    """
    Returns the Hugging Face repository information for the given model repository string.

    Parameters:
        model_repo (str): Repository string in the format 'repo/model-name'.

    Returns:
        dict: Dictionary containing 'repo' and 'name' keys.
    """
    if '/' in model_repo:
        repo, name = model_repo.split('/', 1)
        return {"repo": repo, "name": name}
    else:
        raise ValueError(
            f"Invalid model_repo format: '{model_repo}'. "
            f"Expected format is 'repo/model-name'."
        )
