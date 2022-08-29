
- Most of the code comes from the official library, my purpose is to learn it, and then I add the necessary comments.
- [Reference repository 1 ](https://github.com/facebookresearch/deit)
- [Reference repository 2 ](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models)
- [arxiv](https://arxiv.org/abs/2012.12877)
---
## Notice
Because of the `requirements.txt` , it needs `timm==0.3.2` and `torch==1.7.0` , but in this version , the  `timm.models.layers.helper.py` has a error :
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

it should be :

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
