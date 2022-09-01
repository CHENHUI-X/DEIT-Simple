
- Most of the code comes from the official library, my purpose is to learn it, and then I add the necessary comments.
- [Reference repository 1 : facebookresearch/deit](https://github.com/facebookresearch/deit)
- [Reference repository 2 : timm/models](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models)
- [DEIT : Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)
---
## Notice
Because of the `requirements.txt` , so it recommend to you use a virtual environment , it needs `timm==0.3.2` and `torch==1.7.0` ,  but in this case , if you have an error about `from torch._six import container_abcs' which in `timm.models.layers.helper.py` , you could try changing following code :
```python
    from torch._six import container_abcs
    # From PyTorch internals
    def _ntuple(n):
      def parse(x):
          if isinstance(x, container_abcs.Iterable):
              return x
          return tuple(repeat(x, n))
      return parse
  
```

to

```python 
    import collections.abc
    # From PyTorch internals
    def _ntuple(n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable):
                return x
            return tuple(repeat(x, n))
        return parse

```
Remember finetune it .
