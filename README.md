# numpy_MLP
Implement MLP with only numpy module, without deep learning tools (e.g. PyTorch, Tensorflow, ...)



## Run the code

Simply run ```python ./train.py``` to try our numpy MLP. We perform the MNIST handwritten numbers classification task to conduct the experiment.

You can also use arguments to adjust the training process:

```
e.g.: python ./train.py --optim BGD --param_init kaiming --lr_scheduler const --reg None --more_layers

--epochs: default=50, type=int
--batch_size: default=256, type=int
--optim: default="BGD", type=str, only support "BGD" and "SGD" 
	(BGD: Mini-batch Gradient Descent, SGD: Stochastic Gradient Descent)
--param_init: default="norm", type=str, only support "norm" and "kaiming" 
	(norm: Random normalized initialization, kaiming: Kaiming initialization)
--lr_scheduler: default="const", type=str, only support "const", "multistep" and "exp"
	(const: constant LR, multistep: Multistep decay LR, exp: exponential decay LR)
--reg: default="None", type=str, only support "None", "l1", and "l2" 
	(None: No regularization, l1/l2: L1/L2-Norm regularization)
--more_layers: If tagged in the cmdline, we will add a layer in the MLP model (can't be tagged together with less_layers)
--less_layers: If tagged in the cmdline, we will delete a layer in the MLP model (can't be tagged together with more_layers)

```

Moreover, we've already created 6 sub-directories for 6 different adjustments in the training process. In the sub-directory, we provide learning curves and the corresponding bash file ```run_train.sh```. You can run ```bash run_train.sh``` to directly train the model.

The ```MLP.py``` file is implemented with PyTorch.
