import requests

def get_dataset_info(dataset_id):
    url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
    response = requests.get(url)
    if response.status_code == 200:
        info = response.json()
        print(f"--- {dataset_id} ---")
        try:
            splits = info['dataset_info']['default']['splits']
            for split_name, split_info in splits.items():
                print(f"Split {split_name}: {split_info['num_examples']} examples")
            features = info['dataset_info']['default']['features']
            print("Features:", list(features.keys()))
        except:
            print("Could not parse info")
    else:
        print(f"Failed to fetch {dataset_id}")

if __name__ == "__main__":
    get_dataset_info("theArijitDas/Fake-Reviews-Dataset")
    get_dataset_info("astrosbd/fake-review")
