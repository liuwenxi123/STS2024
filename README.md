# 1. Preprocessing

## 1.1. Data Augmentation

Files within the `preprocessing` directory, such as `augmentation.py`, `rotate.py`, and `sliding_window.py`, can be used to enhance and augment the raw data.

## 1.2. Generating CSV Files

In our project, the network typically retrieves the paths of images and their labels in batches from a CSV file. Therefore, you can slightly modify the `make_data_csv.py` within the `dataset` directory to generate the CSV file needed for running the network.

# 2. Model Prediction

By slightly modifying the data source path in `run_inference(test).py`, you can run it to obtain results in JSON format. There are files such as `json_to_png.py` within the `utils` directory that can facilitate the visualization of the results.
