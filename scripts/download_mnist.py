import os
import urllib.request

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/cvdfoundation/mnist/master/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/cvdfoundation/mnist/master/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://raw.githubusercontent.com/cvdfoundation/mnist/master/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://raw.githubusercontent.com/cvdfoundation/mnist/master/t10k-labels-idx1-ubyte.gz",
}

def download_file(url: str, save_path: str):
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")

def main():
    root = "./data"
    raw_dir = os.path.join(root, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for filename, url in MNIST_URLS.items():
        save_path = os.path.join(raw_dir, filename)
        if os.path.exists(save_path):
            print(f"Skip existing file: {save_path}")
        else:
            download_file(url, save_path)

    print("MNIST raw files are ready.")

if __name__ == "__main__":
    main()
