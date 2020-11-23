# R-MAC Layer for Pytorch

Implementation of R-MAC (Regional Maximum Activations of Convolutions) for Pytorch

Author: Zhijian Qiao

Refer to [TensorFlow2](https://github.com/v-v/RMAC-TensorFlow-2.git)


### Usage:
```python
python demo_torch.py
```
#### Optional Parameters:
* _levels_ - number of levels / scales at which to to generate pooling regions (default = 3)
* _power_ - power exponent to apply (not used by default)
* _overlap_ - overlap percentage between regions (default = 40%)
* _norm_fm_ - normalize feature maps (default = False)
* _sum_fm_ - sum feature maps (default = False)
* _verbose_ - verbose output - shows details about the regions used (default = False)

### Citing:
If you liked and used the code, please consider citing the work and this repository:
```
@article{tolias2016particular,
   author    = {Tolias, Giorgos and Sicre, Ronan and J{\'e}gou, Herv{\'e}},
   title     = {Particular object retrieval with integral max-pooling of CNN activations},
   booktitle = {Proceedings of the International Conference on Learning Representations},
   year      = {2016},
}
```
