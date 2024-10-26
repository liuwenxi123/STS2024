# 1. Preprocessing

## 1.1. Data Augmentation

Files within the `preprocessing` directory, such as `augmentation.py`, `rotate.py`, and `sliding_window.py`, can be used to enhance and augment the raw data.

## 1.2. Generating CSV Files

In our project, the network typically retrieves the paths of images and their labels in batches from a CSV file. Therefore, you can slightly modify the `make_data_csv.py` within the `dataset` directory to generate the CSV file needed for running the network.

# 2. Model Prediction

By slightly modifying the data source path in `run_inference(test).py`, you can run it to obtain results in JSON format. There are files such as `json_to_png.py` within the `utils` directory that can facilitate the visualization of the results.

# 3.  SAM-Med2D Tutorial

Refer to this blog post:[link](https://blog.csdn.net/qq_44886601/article/details/141022007?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522F503FF4B-120B-4F5C-BE64-1A1DE3C145E7%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=F503FF4B-120B-4F5C-BE64-1A1DE3C145E7&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-141022007-null-null.142^v100^pc_search_result_base8&utm_term=sam-med2d%E8%AE%AD%E7%BB%83%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE&spm=1018.2226.3001.4187)
