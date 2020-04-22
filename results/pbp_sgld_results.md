# Experiments and results for SGLD

SGLD params:

- num_nets = 1500
- mix_epochs = 50
- burnin_epochs = 20e3+1
- lr = 5e-5

## Dataset 1

20 points in training set

- x = np.random.uniform(-3.8, 3.8) 
- y = (x^3 + np.random.randn(x.shape[0], x.shape[1])*3)


### Experiment 1

Model details:

- num_hidden = 1
- hidden_units = 25

![Plot](./plots/pbp_sgld_data1_exp1.png)

### Experiment 2

Model details:

- num_hidden = 1
- hidden_units = 50

![Plot](./plots/pbp_sgld_data1_exp2.png)

### Experiment 3

Model details:

- num_hidden = 1
- hidden_units = 100

![Plot](./plots/pbp_sgld_data1_exp3.png)

### Experiment 4

Model details:

- num_hidden = 2
- hidden_units = 25

![Plot](./plots/pbp_sgld_data1_exp4.png)

### Experiment 5

Model details:

- num_hidden = 2
- hidden_units = 50

![Plot](./plots/pbp_sgld_data1_exp5.png)

### Experiment 6

Model details:

- num_hidden = 2
- hidden_units = 100

![Plot](./plots/pbp_sgld_data1_exp6.png)

## Dataset 2

80 points in training set

- x = np.random.uniform(-3.8, 3.8) 
- y = (x^3 + np.random.randn(x.shape[0], x.shape[1])*3)

### Experiment 1

Model details:

- num_hidden = 1
- hidden_units = 25

![Plot](./plots/pbp_sgld_data2_exp1.png)

### Experiment 2

Model details:

- num_hidden = 1
- hidden_units = 50

![Plot](./plots/pbp_sgld_data2_exp2.png)

### Experiment 3

Model details:

- num_hidden = 1
- hidden_units = 100

![Plot](./plots/pbp_sgld_data2_exp3.png)

### Experiment 4

Model details:

- num_hidden = 2
- hidden_units = 25

![Plot](./plots/pbp_sgld_data2_exp4.png)

### Experiment 5

Model details:

- num_hidden = 2
- hidden_units = 50

![Plot](./plots/pbp_sgld_data2_exp5.png)

### Experiment 6

Model details:

- num_hidden = 2
- hidden_units = 100

![Plot](./plots/pbp_sgld_data2_exp6.png)

## Dataset 3

- Using Gaussian distribution with std 1.0
- other params as exp5 of dataset 1

![Plot](./plots/pbp_sgld_data3_exp5.png)

## Dataset 4

- Using Gaussian distribution with std 1.5
- other params as exp5 of dataset 1

![Plot](./plots/pbp_sgld_data4_exp5.png)





