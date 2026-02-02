# <p align="center">Base on BP Neural Network's Temperature forecast</p>
## Introduction
This project is a temperature forecast based on the BP neural network.I just use small dataset to train the model，therefore the model is not very accurate.

If you have better methods to improve the model ablity,you can Through the PR method to contribute to the project.Or you find a new dataset,you can thought to issue tell me.
## Preparatory Work
The code is running in conda environment,therefore you need to install conda first.

Then you should into the path and open the terminal,import the `environment.yml` file.
```bash
conda env create -f environment.yml
```

In the program we will use the selenuim to get the data from the website by using the edge,so you need to install edge and suitable driver.
We can download the edge driver from [Microsoft Edge Tool](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/).Be carefully,you need to install the driver which is suitable for your edge. 

When you successfully download the driver,you need to put the driver into your conda environment.  
If you can't sure successful install the driver,you can use the command to check the driver.
```python
conda activate rt-bp
python -c "from selenium import webdriver; driver = webdriver.Edge(); input('successful install the driver')"
```
Successful display the browser and that not have error,you can close the browser and continue to run the program.

## Running
### Reptile Data
The data is from the website [China Weather](https://www.weather.com.cn/).You can use the command to run the program.
```bash
python reptile.py --area chongqing \
-s ./reptile/output
```
Wait for program to finish,you will see the output file in the directory.
## Training
Point at the datas I choose the BP model to train the data.For a better outcome,I introduce the L2 regularization and Early Stopping to prevent overfitting
```python
model = MLPRegressor(
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=100,
    tol=1e-4 
)
```
The core parameters of MLP were optimized specifically:
```python
model = MLPRegressor(
    hidden_layer_sizes=(100, 50, 25),  
    activation='relu', 
    solver='adam', 
    beta_1=0.9, beta_2=0.999, 
    epsilon=1e-08,  
    max_iter=10000,
)
```
Introduce the relu activation function to prevent the gradient explosion and used adam optimizer to optimize the parameters.

You can use the command to run the program.

```bash
python BPModel.py -r ./reptile/output/pivot
```


## Render
### Line Chart
Using the command to run the program.You can get a forecast result image like this.
```bash
python render.py -r ./forecast
```
<p align="center"><img src="https://github.com/user-attachments/assets/986ff90a-f11d-415b-9fcc-c559e9dc76d2" width="500" height="431"></p>

### Words Cloud
Using the command to run the program.You can get a words cloud like this.

```bash
python wordscloud.py -r ./reptile/output/
```
<p align="center"><img src="https://github.com/user-attachments/assets/4d1e5da7-93cd-446e-8b5c-eb2db54d5658" width="400" height="400"></p>