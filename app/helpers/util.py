from urllib.parse import urlparse

def get_label_by_url(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Get the path part of the URL and split it by '/'
    path_parts = parsed_url.path.strip('/').split('/')

    # Return the second-to-last element, which is the parent directory
    if len(path_parts) >= 2:
        return path_parts[-2]
    return None  # If there is no parent directory