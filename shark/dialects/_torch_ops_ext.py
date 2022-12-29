try:
    # from shark import Tensor, Number
    from torch_mlir.ir import *
    from torch_mlir.dialects._ods_common import (
        get_default_loc_context,
        get_op_result_or_value,
        get_op_results_or_values,
    )
    from ._torch_ops_ext_custom import *
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import List, Optional, Any


class AtenTanhOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTanhOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenTanh_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTanh_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardtanhOp:
    def __init__(self, self_: "shark.Tensor", min_val: "shark.Number", max_val: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min_val):
            min_val = torch_dialect.ConstantNumberOp(min_val)
        else:
            min_val = get_op_result_or_value(min_val)
            assert str(min_val.type) in {'!torch.float', '!torch.int'}, f'min_val should a !torch.number but is {min_val.type}'
            
        if not is_mlir_value(max_val):
            max_val = torch_dialect.ConstantNumberOp(max_val)
        else:
            max_val = get_op_result_or_value(max_val)
            assert str(max_val.type) in {'!torch.float', '!torch.int'}, f'max_val should a !torch.number but is {max_val.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardtanhOp, self).__init__(result_type, self_, min_val, max_val, loc=loc, ip=ip)
        
    
class AtenHardtanh_Op:
    def __init__(self, self_: "shark.Tensor", min_val: "shark.Number", max_val: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min_val):
            min_val = torch_dialect.ConstantNumberOp(min_val)
        else:
            min_val = get_op_result_or_value(min_val)
            assert str(min_val.type) in {'!torch.float', '!torch.int'}, f'min_val should a !torch.number but is {min_val.type}'
            
        if not is_mlir_value(max_val):
            max_val = torch_dialect.ConstantNumberOp(max_val)
        else:
            max_val = get_op_result_or_value(max_val)
            assert str(max_val.type) in {'!torch.float', '!torch.int'}, f'max_val should a !torch.number but is {max_val.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardtanh_Op, self).__init__(result_type, self_, min_val, max_val, loc=loc, ip=ip)
        
    
class AtenReluOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenReluOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRelu_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu6Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRelu6Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRelu6_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRelu6_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLeakyReluOp:
    def __init__(self, self_: "shark.Tensor", negative_slope: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(negative_slope):
            negative_slope = torch_dialect.ConstantNumberOp(negative_slope)
        else:
            negative_slope = get_op_result_or_value(negative_slope)
            assert str(negative_slope.type) in {'!torch.float', '!torch.int'}, f'negative_slope should a !torch.number but is {negative_slope.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLeakyReluOp, self).__init__(result_type, self_, negative_slope, loc=loc, ip=ip)
        
    
class AtenLeakyRelu_Op:
    def __init__(self, self_: "shark.Tensor", negative_slope: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(negative_slope):
            negative_slope = torch_dialect.ConstantNumberOp(negative_slope)
        else:
            negative_slope = get_op_result_or_value(negative_slope)
            assert str(negative_slope.type) in {'!torch.float', '!torch.int'}, f'negative_slope should a !torch.number but is {negative_slope.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLeakyRelu_Op, self).__init__(result_type, self_, negative_slope, loc=loc, ip=ip)
        
    
class AtenLogOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLog_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSigmoidOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSigmoidOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSigmoid_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSigmoid_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardsigmoidOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardsigmoidOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardsigmoid_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardsigmoid_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardswishOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardswishOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenHardswish_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenHardswish_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenErfOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenErfOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenErf_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenErf_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSiluOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSiluOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSilu_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSilu_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSinOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSinOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSin_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSin_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExp_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExp_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpm1Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpm1Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenExpm1_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpm1_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCosOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCosOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCos_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCos_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAtan2Op:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAtan2Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAtan2_Op:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAtan2_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNegOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNegOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenNeg_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNeg_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFloorOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFloorOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFloor_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFloor_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCeilOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCeilOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenCeil_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCeil_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseNotOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseNotOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseNot_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseNot_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenDivTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDivTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDiv_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDiv_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalOrOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalOrOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalOr_Op:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalOr_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalAndOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalAndOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalAnd_Op:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalAnd_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalXorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalXorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalXor_Op:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalXor_Op, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogicalNotOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalNotOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLogicalNot_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogicalNot_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLerpTensorOp:
    def __init__(self, self_: "shark.Tensor", end: "shark.Tensor", weight: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(end):
            assert is_mlir_value(end), f'end should be a Tensor but is {end.type}'
        else:
            end = get_op_result_or_value(end)
            assert str(end.type).startswith("!torch.vtensor"), f'end should a Tensor but is {end.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLerpTensorOp, self).__init__(result_type, self_, end, weight, loc=loc, ip=ip)
        
    
class AtenLerp_TensorOp:
    def __init__(self, self_: "shark.Tensor", end: "shark.Tensor", weight: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(end):
            assert is_mlir_value(end), f'end should be a Tensor but is {end.type}'
        else:
            end = get_op_result_or_value(end)
            assert str(end.type).startswith("!torch.vtensor"), f'end should a Tensor but is {end.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLerp_TensorOp, self).__init__(result_type, self_, end, weight, loc=loc, ip=ip)
        
    
class AtenEqTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEqTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEq_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEq_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGtTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGtTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGt_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGt_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGeTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGe_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLtTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLtTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLt_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLt_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLeTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLe_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNeTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNeTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNe_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNe_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDivScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDivScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenDiv_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDiv_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNeScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenNe_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEqScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEqScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenEq_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEq_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGtScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGtScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGt_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGt_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGeScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenGe_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLtScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLtScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLt_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLt_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLeScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLeScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLe_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLe_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenFmodScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFmodScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenFmod_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFmod_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMaskedFillScalarOp:
    def __init__(self, self_: "shark.Tensor", mask: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaskedFillScalarOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFill_ScalarOp:
    def __init__(self, self_: "shark.Tensor", mask: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaskedFill_ScalarOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFillTensorOp:
    def __init__(self, self_: "shark.Tensor", mask: "shark.Tensor", value: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'value should be a Tensor but is {value.type}'
        else:
            value = get_op_result_or_value(value)
            assert str(value.type).startswith("!torch.vtensor"), f'value should a Tensor but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaskedFillTensorOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenMaskedFill_TensorOp:
    def __init__(self, self_: "shark.Tensor", mask: "shark.Tensor", value: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'value should be a Tensor but is {value.type}'
        else:
            value = get_op_result_or_value(value)
            assert str(value.type).startswith("!torch.vtensor"), f'value should a Tensor but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaskedFill_TensorOp, self).__init__(result_type, self_, mask, value, loc=loc, ip=ip)
        
    
class AtenClampOp:
    def __init__(self, self_: "shark.Tensor", min: Optional["shark.Number"], max: Optional["shark.Number"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min):
            if min is not None:
                min = torch_dialect.ConstantNumberOp(min)
            else:
                min = torch_dialect.ConstantNoneOp()
        else:
            min = get_op_result_or_value(min)
            assert str(min.type) in {'!torch.float', '!torch.int'}, f'min should a !torch.number but is {min.type}'
            
        if not is_mlir_value(max):
            if max is not None:
                max = torch_dialect.ConstantNumberOp(max)
            else:
                max = torch_dialect.ConstantNoneOp()
        else:
            max = get_op_result_or_value(max)
            assert str(max.type) in {'!torch.float', '!torch.int'}, f'max should a !torch.number but is {max.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClampOp, self).__init__(result_type, self_, min, max, loc=loc, ip=ip)
        
    
class AtenClamp_Op:
    def __init__(self, self_: "shark.Tensor", min: Optional["shark.Number"], max: Optional["shark.Number"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min):
            if min is not None:
                min = torch_dialect.ConstantNumberOp(min)
            else:
                min = torch_dialect.ConstantNoneOp()
        else:
            min = get_op_result_or_value(min)
            assert str(min.type) in {'!torch.float', '!torch.int'}, f'min should a !torch.number but is {min.type}'
            
        if not is_mlir_value(max):
            if max is not None:
                max = torch_dialect.ConstantNumberOp(max)
            else:
                max = torch_dialect.ConstantNoneOp()
        else:
            max = get_op_result_or_value(max)
            assert str(max.type) in {'!torch.float', '!torch.int'}, f'max should a !torch.number but is {max.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClamp_Op, self).__init__(result_type, self_, min, max, loc=loc, ip=ip)
        
    
class AtenClampMinOp:
    def __init__(self, self_: "shark.Tensor", min: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min):
            min = torch_dialect.ConstantNumberOp(min)
        else:
            min = get_op_result_or_value(min)
            assert str(min.type) in {'!torch.float', '!torch.int'}, f'min should a !torch.number but is {min.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClampMinOp, self).__init__(result_type, self_, min, loc=loc, ip=ip)
        
    
class AtenClampMin_Op:
    def __init__(self, self_: "shark.Tensor", min: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(min):
            min = torch_dialect.ConstantNumberOp(min)
        else:
            min = get_op_result_or_value(min)
            assert str(min.type) in {'!torch.float', '!torch.int'}, f'min should a !torch.number but is {min.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClampMin_Op, self).__init__(result_type, self_, min, loc=loc, ip=ip)
        
    
class AtenClampMaxOp:
    def __init__(self, self_: "shark.Tensor", max: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(max):
            max = torch_dialect.ConstantNumberOp(max)
        else:
            max = get_op_result_or_value(max)
            assert str(max.type) in {'!torch.float', '!torch.int'}, f'max should a !torch.number but is {max.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClampMaxOp, self).__init__(result_type, self_, max, loc=loc, ip=ip)
        
    
class AtenClampMax_Op:
    def __init__(self, self_: "shark.Tensor", max: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(max):
            max = torch_dialect.ConstantNumberOp(max)
        else:
            max = get_op_result_or_value(max)
            assert str(max.type) in {'!torch.float', '!torch.int'}, f'max should a !torch.number but is {max.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenClampMax_Op, self).__init__(result_type, self_, max, loc=loc, ip=ip)
        
    
class AtenLog2Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLog2Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog2_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLog2_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqrtOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqrtOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqrt_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqrt_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog1pOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLog1pOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenLog1p_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLog1p_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsqrtOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRsqrtOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsqrt_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRsqrt_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAbsOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAbsOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAbs_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAbs_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenReciprocalOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenReciprocalOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenReciprocal_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenReciprocal_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBitwiseAndTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseAndTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseAnd_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseAnd_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseOrTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseOrTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBitwiseOr_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBitwiseOr_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenThresholdOp:
    def __init__(self, self_: "shark.Tensor", threshold: "shark.Number", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert str(threshold.type) in {'!torch.float', '!torch.int'}, f'threshold should a !torch.number but is {threshold.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenThresholdOp, self).__init__(result_type, self_, threshold, value, loc=loc, ip=ip)
        
    
class AtenThreshold_Op:
    def __init__(self, self_: "shark.Tensor", threshold: "shark.Number", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert str(threshold.type) in {'!torch.float', '!torch.int'}, f'threshold should a !torch.number but is {threshold.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenThreshold_Op, self).__init__(result_type, self_, threshold, value, loc=loc, ip=ip)
        
    
class AtenSquareOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSquareOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSquare_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSquare_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenUnsqueezeOp:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUnsqueezeOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenUnsqueeze_Op:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUnsqueeze_Op, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenZeroOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenZeroOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenZero_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenZero_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFillScalarOp:
    def __init__(self, self_: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFillScalarOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFill_ScalarOp:
    def __init__(self, self_: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFill_ScalarOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFillTensorOp:
    def __init__(self, self_: "shark.Tensor", value: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'value should be a Tensor but is {value.type}'
        else:
            value = get_op_result_or_value(value)
            assert str(value.type).startswith("!torch.vtensor"), f'value should a Tensor but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFillTensorOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenFill_TensorOp:
    def __init__(self, self_: "shark.Tensor", value: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(value):
            assert is_mlir_value(value), f'value should be a Tensor but is {value.type}'
        else:
            value = get_op_result_or_value(value)
            assert str(value.type).startswith("!torch.vtensor"), f'value should a Tensor but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFill_TensorOp, self).__init__(result_type, self_, value, loc=loc, ip=ip)
        
    
class AtenDivTensorModeOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", rounding_mode: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(rounding_mode):
            if rounding_mode is not None:
                rounding_mode = torch_dialect.ConstantStrOp(rounding_mode)
            else:
                rounding_mode = torch_dialect.ConstantNoneOp()
        else:
            rounding_mode = get_op_result_or_value(rounding_mode)
            assert str(rounding_mode.type) == '!torch.str', f'rounding_mode should a !torch.str but is {rounding_mode.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDivTensorModeOp, self).__init__(result_type, self_, other, rounding_mode, loc=loc, ip=ip)
        
    
class AtenDiv_TensorModeOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", rounding_mode: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(rounding_mode):
            if rounding_mode is not None:
                rounding_mode = torch_dialect.ConstantStrOp(rounding_mode)
            else:
                rounding_mode = torch_dialect.ConstantNoneOp()
        else:
            rounding_mode = get_op_result_or_value(rounding_mode)
            assert str(rounding_mode.type) == '!torch.str', f'rounding_mode should a !torch.str but is {rounding_mode.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDiv_TensorModeOp, self).__init__(result_type, self_, other, rounding_mode, loc=loc, ip=ip)
        
    
class AtenMulTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMulTensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMul_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMul_TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddTensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAdd_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAdd_TensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSubTensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSubTensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSub_TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSub_TensorOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAddScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenAdd_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAdd_ScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSubScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSubScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenSub_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSub_ScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenMulScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMulScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMul_ScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMul_ScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddcmulOp:
    def __init__(self, self_: "shark.Tensor", tensor1: "shark.Tensor", tensor2: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'tensor1 should be a Tensor but is {tensor1.type}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert str(tensor1.type).startswith("!torch.vtensor"), f'tensor1 should a Tensor but is {tensor1.type}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'tensor2 should be a Tensor but is {tensor2.type}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert str(tensor2.type).startswith("!torch.vtensor"), f'tensor2 should a Tensor but is {tensor2.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddcmulOp, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcmul_Op:
    def __init__(self, self_: "shark.Tensor", tensor1: "shark.Tensor", tensor2: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'tensor1 should be a Tensor but is {tensor1.type}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert str(tensor1.type).startswith("!torch.vtensor"), f'tensor1 should a Tensor but is {tensor1.type}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'tensor2 should be a Tensor but is {tensor2.type}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert str(tensor2.type).startswith("!torch.vtensor"), f'tensor2 should a Tensor but is {tensor2.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddcmul_Op, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcdivOp:
    def __init__(self, self_: "shark.Tensor", tensor1: "shark.Tensor", tensor2: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'tensor1 should be a Tensor but is {tensor1.type}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert str(tensor1.type).startswith("!torch.vtensor"), f'tensor1 should a Tensor but is {tensor1.type}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'tensor2 should be a Tensor but is {tensor2.type}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert str(tensor2.type).startswith("!torch.vtensor"), f'tensor2 should a Tensor but is {tensor2.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddcdivOp, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenAddcdiv_Op:
    def __init__(self, self_: "shark.Tensor", tensor1: "shark.Tensor", tensor2: "shark.Tensor", value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(tensor1):
            assert is_mlir_value(tensor1), f'tensor1 should be a Tensor but is {tensor1.type}'
        else:
            tensor1 = get_op_result_or_value(tensor1)
            assert str(tensor1.type).startswith("!torch.vtensor"), f'tensor1 should a Tensor but is {tensor1.type}'
            
        if not is_mlir_value(tensor2):
            assert is_mlir_value(tensor2), f'tensor2 should be a Tensor but is {tensor2.type}'
        else:
            tensor2 = get_op_result_or_value(tensor2)
            assert str(tensor2.type).startswith("!torch.vtensor"), f'tensor2 should a Tensor but is {tensor2.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddcdiv_Op, self).__init__(result_type, self_, tensor1, tensor2, value, loc=loc, ip=ip)
        
    
class AtenMaximumOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaximumOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMinimumOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMinimumOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMishOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMishOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRsubScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRsubScalarOp, self).__init__(result_type, self_, other, alpha, loc=loc, ip=ip)
        
    
class AtenGeluOp:
    def __init__(self, self_: "shark.Tensor", approximate: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(approximate):
            approximate = torch_dialect.ConstantStrOp(approximate)
        else:
            approximate = get_op_result_or_value(approximate)
            assert str(approximate.type) == '!torch.str', f'approximate should a !torch.str but is {approximate.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGeluOp, self).__init__(result_type, self_, approximate, loc=loc, ip=ip)
        
    
class AtenPowTensorScalarOp:
    def __init__(self, self_: "shark.Tensor", exponent: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(exponent):
            exponent = torch_dialect.ConstantNumberOp(exponent)
        else:
            exponent = get_op_result_or_value(exponent)
            assert str(exponent.type) in {'!torch.float', '!torch.int'}, f'exponent should a !torch.number but is {exponent.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenPowTensorScalarOp, self).__init__(result_type, self_, exponent, loc=loc, ip=ip)
        
    
class AtenPowTensorTensorOp:
    def __init__(self, self_: "shark.Tensor", exponent: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(exponent):
            assert is_mlir_value(exponent), f'exponent should be a Tensor but is {exponent.type}'
        else:
            exponent = get_op_result_or_value(exponent)
            assert str(exponent.type).startswith("!torch.vtensor"), f'exponent should a Tensor but is {exponent.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenPowTensorTensorOp, self).__init__(result_type, self_, exponent, loc=loc, ip=ip)
        
    
class AtenThresholdBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", self_: "shark.Tensor", threshold: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert str(threshold.type) in {'!torch.float', '!torch.int'}, f'threshold should a !torch.number but is {threshold.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenThresholdBackwardOp, self).__init__(result_type, grad_output, self_, threshold, loc=loc, ip=ip)
        
    
class AtenFloorDivideOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFloorDivideOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenSoftplusOp:
    def __init__(self, self_: "shark.Tensor", beta: "shark.Number", threshold: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert str(beta.type) in {'!torch.float', '!torch.int'}, f'beta should a !torch.number but is {beta.type}'
            
        if not is_mlir_value(threshold):
            threshold = torch_dialect.ConstantNumberOp(threshold)
        else:
            threshold = get_op_result_or_value(threshold)
            assert str(threshold.type) in {'!torch.float', '!torch.int'}, f'threshold should a !torch.number but is {threshold.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSoftplusOp, self).__init__(result_type, self_, beta, threshold, loc=loc, ip=ip)
        
    
class AtenTriuOp:
    def __init__(self, self_: "shark.Tensor", diagonal: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(diagonal):
            diagonal = torch_dialect.ConstantIntOp(diagonal)
        else:
            diagonal = get_op_result_or_value(diagonal)
            assert str(diagonal.type) == '!torch.int', f'diagonal should a !torch.int but is {diagonal.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTriuOp, self).__init__(result_type, self_, diagonal, loc=loc, ip=ip)
        
    
class AtenTriu_Op:
    def __init__(self, self_: "shark.Tensor", diagonal: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(diagonal):
            diagonal = torch_dialect.ConstantIntOp(diagonal)
        else:
            diagonal = get_op_result_or_value(diagonal)
            assert str(diagonal.type) == '!torch.int', f'diagonal should a !torch.int but is {diagonal.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTriu_Op, self).__init__(result_type, self_, diagonal, loc=loc, ip=ip)
        
    
class AtenRoundOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRoundOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenRound_Op:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRound_Op, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenIndexPutHackedTwinOp:
    def __init__(self, self_: "shark.Tensor", indices: List["shark.Tensor"], values: "shark.Tensor", accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type) == '!torch.list<Tensor>', f'indices should a !torch.list<Tensor> but is {indices.type}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'values should be a Tensor but is {values.type}'
        else:
            values = get_op_result_or_value(values)
            assert str(values.type).startswith("!torch.vtensor"), f'values should a Tensor but is {values.type}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert str(accumulate.type) == '!torch.bool', f'accumulate should a !torch.bool but is {accumulate.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenIndexPutHackedTwinOp, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenIndexPut_HackedTwinOp:
    def __init__(self, self_: "shark.Tensor", indices: List["shark.Tensor"], values: "shark.Tensor", accumulate: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type) == '!torch.list<Tensor>', f'indices should a !torch.list<Tensor> but is {indices.type}'
            
        if not is_mlir_value(values):
            assert is_mlir_value(values), f'values should be a Tensor but is {values.type}'
        else:
            values = get_op_result_or_value(values)
            assert str(values.type).startswith("!torch.vtensor"), f'values should a Tensor but is {values.type}'
            
        if not is_mlir_value(accumulate):
            accumulate = torch_dialect.ConstantBoolOp(accumulate)
        else:
            accumulate = get_op_result_or_value(accumulate)
            assert str(accumulate.type) == '!torch.bool', f'accumulate should a !torch.bool but is {accumulate.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenIndexPut_HackedTwinOp, self).__init__(result_type, self_, indices, values, accumulate, loc=loc, ip=ip)
        
    
class AtenLinearOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLinearOp, self).__init__(result_type, input, weight, bias, loc=loc, ip=ip)
        
    
class AtenMmOp:
    def __init__(self, self_: "shark.Tensor", mat2: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'mat2 should be a Tensor but is {mat2.type}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert str(mat2.type).startswith("!torch.vtensor"), f'mat2 should a Tensor but is {mat2.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)
        
    
class AtenAddmmOp:
    def __init__(self, self_: "shark.Tensor", mat1: "shark.Tensor", mat2: "shark.Tensor", beta: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mat1):
            assert is_mlir_value(mat1), f'mat1 should be a Tensor but is {mat1.type}'
        else:
            mat1 = get_op_result_or_value(mat1)
            assert str(mat1.type).startswith("!torch.vtensor"), f'mat1 should a Tensor but is {mat1.type}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'mat2 should be a Tensor but is {mat2.type}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert str(mat2.type).startswith("!torch.vtensor"), f'mat2 should a Tensor but is {mat2.type}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert str(beta.type) in {'!torch.float', '!torch.int'}, f'beta should a !torch.number but is {beta.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAddmmOp, self).__init__(result_type, self_, mat1, mat2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenMatmulOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMatmulOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenMvOp:
    def __init__(self, self_: "shark.Tensor", vec: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(vec):
            assert is_mlir_value(vec), f'vec should be a Tensor but is {vec.type}'
        else:
            vec = get_op_result_or_value(vec)
            assert str(vec.type).startswith("!torch.vtensor"), f'vec should a Tensor but is {vec.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMvOp, self).__init__(result_type, self_, vec, loc=loc, ip=ip)
        
    
class AtenConv2dOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], dilation: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConv2dOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, groups, loc=loc, ip=ip)
        
    
class AtenConvTranspose1dOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConvTranspose1dOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvTranspose2dInputOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConvTranspose2dInputOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvTranspose3dInputOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], output_padding: List[int], groups: int, dilation: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConvTranspose3dInputOp, self).__init__(result_type, input, weight, bias, stride, padding, output_padding, groups, dilation, loc=loc, ip=ip)
        
    
class AtenConvolutionOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert str(transposed.type) == '!torch.bool', f'transposed should a !torch.bool but is {transposed.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConvolutionOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, loc=loc, ip=ip)
        
    
class AtenConvolutionOverrideableOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert str(transposed.type) == '!torch.bool', f'transposed should a !torch.bool but is {transposed.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConvolutionOverrideableOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, loc=loc, ip=ip)
        
    
class Aten_ConvolutionOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, allow_tf32: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert str(transposed.type) == '!torch.bool', f'transposed should a !torch.bool but is {transposed.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(benchmark):
            benchmark = torch_dialect.ConstantBoolOp(benchmark)
        else:
            benchmark = get_op_result_or_value(benchmark)
            assert str(benchmark.type) == '!torch.bool', f'benchmark should a !torch.bool but is {benchmark.type}'
            
        if not is_mlir_value(deterministic):
            deterministic = torch_dialect.ConstantBoolOp(deterministic)
        else:
            deterministic = get_op_result_or_value(deterministic)
            assert str(deterministic.type) == '!torch.bool', f'deterministic should a !torch.bool but is {deterministic.type}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert str(cudnn_enabled.type) == '!torch.bool', f'cudnn_enabled should a !torch.bool but is {cudnn_enabled.type}'
            
        if not is_mlir_value(allow_tf32):
            allow_tf32 = torch_dialect.ConstantBoolOp(allow_tf32)
        else:
            allow_tf32 = get_op_result_or_value(allow_tf32)
            assert str(allow_tf32.type) == '!torch.bool', f'allow_tf32 should a !torch.bool but is {allow_tf32.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_ConvolutionOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, allow_tf32, loc=loc, ip=ip)
        
    
class Aten_ConvolutionDeprecatedOp:
    def __init__(self, input: "shark.Tensor", weight: "shark.Tensor", bias: Optional["shark.Tensor"], stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, benchmark: bool, deterministic: bool, cudnn_enabled: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert str(transposed.type) == '!torch.bool', f'transposed should a !torch.bool but is {transposed.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(benchmark):
            benchmark = torch_dialect.ConstantBoolOp(benchmark)
        else:
            benchmark = get_op_result_or_value(benchmark)
            assert str(benchmark.type) == '!torch.bool', f'benchmark should a !torch.bool but is {benchmark.type}'
            
        if not is_mlir_value(deterministic):
            deterministic = torch_dialect.ConstantBoolOp(deterministic)
        else:
            deterministic = get_op_result_or_value(deterministic)
            assert str(deterministic.type) == '!torch.bool', f'deterministic should a !torch.bool but is {deterministic.type}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert str(cudnn_enabled.type) == '!torch.bool', f'cudnn_enabled should a !torch.bool but is {cudnn_enabled.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_ConvolutionDeprecatedOp, self).__init__(result_type, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, loc=loc, ip=ip)
        
    
class AtenRollOp:
    def __init__(self, self_: "shark.Tensor", shifts: List[int], dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(shifts):
            shifts = list(map(torch_dialect.ConstantIntOp, shifts))
            shifts = torch_dialect.PrimListConstructOp(shifts)
        else:
            shifts = get_op_result_or_value(shifts)
            assert str(shifts.type) == '!torch.list<int>', f'shifts should a !torch.list<int> but is {shifts.type}'
            
        if not is_mlir_value(dims):
            dims = list(map(torch_dialect.ConstantIntOp, dims))
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert str(dims.type) == '!torch.list<int>', f'dims should a !torch.list<int> but is {dims.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRollOp, self).__init__(result_type, self_, shifts, dims, loc=loc, ip=ip)
        
    
class AtenConvolutionBackwardOverrideableOp:
    def __init__(self, grad_output: "shark.Tensor", input: "shark.Tensor", weight: "shark.Tensor", stride: List[int], padding: List[int], dilation: List[int], transposed: bool, output_padding: List[int], groups: int, output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(transposed):
            transposed = torch_dialect.ConstantBoolOp(transposed)
        else:
            transposed = get_op_result_or_value(transposed)
            assert str(transposed.type) == '!torch.bool', f'transposed should a !torch.bool but is {transposed.type}'
            
        if not is_mlir_value(output_padding):
            output_padding = list(map(torch_dialect.ConstantIntOp, output_padding))
            output_padding = torch_dialect.PrimListConstructOp(output_padding)
        else:
            output_padding = get_op_result_or_value(output_padding)
            assert str(output_padding.type) == '!torch.list<int>', f'output_padding should a !torch.list<int> but is {output_padding.type}'
            
        if not is_mlir_value(groups):
            groups = torch_dialect.ConstantIntOp(groups)
        else:
            groups = get_op_result_or_value(groups)
            assert str(groups.type) == '!torch.int', f'groups should a !torch.int but is {groups.type}'
            
        if not is_mlir_value(output_mask):
            output_mask = list(map(torch_dialect.ConstantBoolOp, output_mask))
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            # should be bool[]
            pass
            
        grad_input_type = Type.parse("!torch.vtensor")
        grad_weight_type = Type.parse("!torch.vtensor")
        grad_bias_type = Type.parse("!torch.vtensor")
        super(AtenConvolutionBackwardOverrideableOp, self).__init__(grad_input_type, grad_weight_type, grad_bias_type, grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask, loc=loc, ip=ip)
        
    
class AtenFlipOp:
    def __init__(self, self_: "shark.Tensor", dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dims):
            dims = list(map(torch_dialect.ConstantIntOp, dims))
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert str(dims.type) == '!torch.list<int>', f'dims should a !torch.list<int> but is {dims.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFlipOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class AtenNativeBatchNormOp:
    def __init__(self, input: "shark.Tensor", weight: Optional["shark.Tensor"], bias: Optional["shark.Tensor"], running_mean: Optional["shark.Tensor"], running_var: Optional["shark.Tensor"], training: bool, momentum: float, eps: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'running_mean should be a Tensor but is {running_mean.type}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert str(running_mean.type).startswith("!torch.vtensor"), f'running_mean should a Tensor but is {running_mean.type}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'running_var should be a Tensor but is {running_var.type}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert str(running_var.type).startswith("!torch.vtensor"), f'running_var should a Tensor but is {running_var.type}'
            
        if not is_mlir_value(training):
            training = torch_dialect.ConstantBoolOp(training)
        else:
            training = get_op_result_or_value(training)
            assert str(training.type) == '!torch.bool', f'training should a !torch.bool but is {training.type}'
            
        if not is_mlir_value(momentum):
            momentum = torch_dialect.ConstantFloatOp(momentum)
        else:
            momentum = get_op_result_or_value(momentum)
            assert str(momentum.type) == '!torch.float', f'momentum should a !torch.float but is {momentum.type}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert str(eps.type) == '!torch.float', f'eps should a !torch.float but is {eps.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        super(AtenNativeBatchNormOp, self).__init__(result0_type, result1_type, result2_type, input, weight, bias, running_mean, running_var, training, momentum, eps, loc=loc, ip=ip)
        
    
class AtenBatchNormOp:
    def __init__(self, input: "shark.Tensor", weight: Optional["shark.Tensor"], bias: Optional["shark.Tensor"], running_mean: Optional["shark.Tensor"], running_var: Optional["shark.Tensor"], training: bool, momentum: float, eps: float, cudnn_enabled: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'running_mean should be a Tensor but is {running_mean.type}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert str(running_mean.type).startswith("!torch.vtensor"), f'running_mean should a Tensor but is {running_mean.type}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'running_var should be a Tensor but is {running_var.type}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert str(running_var.type).startswith("!torch.vtensor"), f'running_var should a Tensor but is {running_var.type}'
            
        if not is_mlir_value(training):
            training = torch_dialect.ConstantBoolOp(training)
        else:
            training = get_op_result_or_value(training)
            assert str(training.type) == '!torch.bool', f'training should a !torch.bool but is {training.type}'
            
        if not is_mlir_value(momentum):
            momentum = torch_dialect.ConstantFloatOp(momentum)
        else:
            momentum = get_op_result_or_value(momentum)
            assert str(momentum.type) == '!torch.float', f'momentum should a !torch.float but is {momentum.type}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert str(eps.type) == '!torch.float', f'eps should a !torch.float but is {eps.type}'
            
        if not is_mlir_value(cudnn_enabled):
            cudnn_enabled = torch_dialect.ConstantBoolOp(cudnn_enabled)
        else:
            cudnn_enabled = get_op_result_or_value(cudnn_enabled)
            assert str(cudnn_enabled.type) == '!torch.bool', f'cudnn_enabled should a !torch.bool but is {cudnn_enabled.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBatchNormOp, self).__init__(result_type, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled, loc=loc, ip=ip)
        
    
class AtenLayerNormOp:
    def __init__(self, input: "shark.Tensor", normalized_shape: List[int], weight: Optional["shark.Tensor"], bias: Optional["shark.Tensor"], eps: float, cudnn_enable: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = list(map(torch_dialect.ConstantIntOp, normalized_shape))
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert str(normalized_shape.type) == '!torch.list<int>', f'normalized_shape should a !torch.list<int> but is {normalized_shape.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert str(eps.type) == '!torch.float', f'eps should a !torch.float but is {eps.type}'
            
        if not is_mlir_value(cudnn_enable):
            cudnn_enable = torch_dialect.ConstantBoolOp(cudnn_enable)
        else:
            cudnn_enable = get_op_result_or_value(cudnn_enable)
            assert str(cudnn_enable.type) == '!torch.bool', f'cudnn_enable should a !torch.bool but is {cudnn_enable.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLayerNormOp, self).__init__(result_type, input, normalized_shape, weight, bias, eps, cudnn_enable, loc=loc, ip=ip)
        
    
class AtenNativeLayerNormOp:
    def __init__(self, input: "shark.Tensor", normalized_shape: List[int], weight: Optional["shark.Tensor"], bias: Optional["shark.Tensor"], eps: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = list(map(torch_dialect.ConstantIntOp, normalized_shape))
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert str(normalized_shape.type) == '!torch.list<int>', f'normalized_shape should a !torch.list<int> but is {normalized_shape.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert str(eps.type) == '!torch.float', f'eps should a !torch.float but is {eps.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        super(AtenNativeLayerNormOp, self).__init__(result0_type, result1_type, result2_type, input, normalized_shape, weight, bias, eps, loc=loc, ip=ip)
        
    
class AtenMaxPool2dOp:
    def __init__(self, self_: "shark.Tensor", kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = list(map(torch_dialect.ConstantIntOp, kernel_size))
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert str(kernel_size.type) == '!torch.list<int>', f'kernel_size should a !torch.list<int> but is {kernel_size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert str(ceil_mode.type) == '!torch.bool', f'ceil_mode should a !torch.bool but is {ceil_mode.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaxPool2dOp, self).__init__(result_type, self_, kernel_size, stride, padding, dilation, ceil_mode, loc=loc, ip=ip)
        
    
class AtenMaxPool2dWithIndicesOp:
    def __init__(self, self_: "shark.Tensor", kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = list(map(torch_dialect.ConstantIntOp, kernel_size))
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert str(kernel_size.type) == '!torch.list<int>', f'kernel_size should a !torch.list<int> but is {kernel_size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert str(ceil_mode.type) == '!torch.bool', f'ceil_mode should a !torch.bool but is {ceil_mode.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        super(AtenMaxPool2dWithIndicesOp, self).__init__(result0_type, result1_type, self_, kernel_size, stride, padding, dilation, ceil_mode, loc=loc, ip=ip)
        
    
class AtenMaxPool2dWithIndicesBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", self_: "shark.Tensor", kernel_size: List[int], stride: List[int], padding: List[int], dilation: List[int], ceil_mode: bool, indices: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = list(map(torch_dialect.ConstantIntOp, kernel_size))
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert str(kernel_size.type) == '!torch.list<int>', f'kernel_size should a !torch.list<int> but is {kernel_size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(dilation):
            dilation = list(map(torch_dialect.ConstantIntOp, dilation))
            dilation = torch_dialect.PrimListConstructOp(dilation)
        else:
            dilation = get_op_result_or_value(dilation)
            assert str(dilation.type) == '!torch.list<int>', f'dilation should a !torch.list<int> but is {dilation.type}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert str(ceil_mode.type) == '!torch.bool', f'ceil_mode should a !torch.bool but is {ceil_mode.type}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'indices should be a Tensor but is {indices.type}'
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type).startswith("!torch.vtensor"), f'indices should a Tensor but is {indices.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaxPool2dWithIndicesBackwardOp, self).__init__(result_type, grad_output, self_, kernel_size, stride, padding, dilation, ceil_mode, indices, loc=loc, ip=ip)
        
    
class AtenAvgPool2dOp:
    def __init__(self, self_: "shark.Tensor", kernel_size: List[int], stride: List[int], padding: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(kernel_size):
            kernel_size = list(map(torch_dialect.ConstantIntOp, kernel_size))
            kernel_size = torch_dialect.PrimListConstructOp(kernel_size)
        else:
            kernel_size = get_op_result_or_value(kernel_size)
            assert str(kernel_size.type) == '!torch.list<int>', f'kernel_size should a !torch.list<int> but is {kernel_size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(padding):
            padding = list(map(torch_dialect.ConstantIntOp, padding))
            padding = torch_dialect.PrimListConstructOp(padding)
        else:
            padding = get_op_result_or_value(padding)
            assert str(padding.type) == '!torch.list<int>', f'padding should a !torch.list<int> but is {padding.type}'
            
        if not is_mlir_value(ceil_mode):
            ceil_mode = torch_dialect.ConstantBoolOp(ceil_mode)
        else:
            ceil_mode = get_op_result_or_value(ceil_mode)
            assert str(ceil_mode.type) == '!torch.bool', f'ceil_mode should a !torch.bool but is {ceil_mode.type}'
            
        if not is_mlir_value(count_include_pad):
            count_include_pad = torch_dialect.ConstantBoolOp(count_include_pad)
        else:
            count_include_pad = get_op_result_or_value(count_include_pad)
            assert str(count_include_pad.type) == '!torch.bool', f'count_include_pad should a !torch.bool but is {count_include_pad.type}'
            
        if not is_mlir_value(divisor_override):
            if divisor_override is not None:
                divisor_override = torch_dialect.ConstantIntOp(divisor_override)
            else:
                divisor_override = torch_dialect.ConstantNoneOp()
        else:
            divisor_override = get_op_result_or_value(divisor_override)
            assert str(divisor_override.type) == '!torch.int', f'divisor_override should a !torch.int but is {divisor_override.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAvgPool2dOp, self).__init__(result_type, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, loc=loc, ip=ip)
        
    
class AtenSoftmaxIntOp:
    def __init__(self, self_: "shark.Tensor", dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSoftmaxIntOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class AtenLogSoftmaxIntOp:
    def __init__(self, self_: "shark.Tensor", dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogSoftmaxIntOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class Aten_LogSoftmaxOp:
    def __init__(self, self_: "shark.Tensor", dim: int, half_to_float: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(half_to_float):
            half_to_float = torch_dialect.ConstantBoolOp(half_to_float)
        else:
            half_to_float = get_op_result_or_value(half_to_float)
            assert str(half_to_float.type) == '!torch.bool', f'half_to_float should a !torch.bool but is {half_to_float.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_LogSoftmaxOp, self).__init__(result_type, self_, dim, half_to_float, loc=loc, ip=ip)
        
    
class AtenAdaptiveAvgPool2dOp:
    def __init__(self, self_: "shark.Tensor", output_size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(output_size):
            output_size = list(map(torch_dialect.ConstantIntOp, output_size))
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert str(output_size.type) == '!torch.list<int>', f'output_size should a !torch.list<int> but is {output_size.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAdaptiveAvgPool2dOp, self).__init__(result_type, self_, output_size, loc=loc, ip=ip)
        
    
class AtenTopkOp:
    def __init__(self, self_: "shark.Tensor", k: int, dim: int, largest: bool, sorted: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(k):
            k = torch_dialect.ConstantIntOp(k)
        else:
            k = get_op_result_or_value(k)
            assert str(k.type) == '!torch.int', f'k should a !torch.int but is {k.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(largest):
            largest = torch_dialect.ConstantBoolOp(largest)
        else:
            largest = get_op_result_or_value(largest)
            assert str(largest.type) == '!torch.bool', f'largest should a !torch.bool but is {largest.type}'
            
        if not is_mlir_value(sorted):
            sorted = torch_dialect.ConstantBoolOp(sorted)
        else:
            sorted = get_op_result_or_value(sorted)
            assert str(sorted.type) == '!torch.bool', f'sorted should a !torch.bool but is {sorted.type}'
            
        values_type = Type.parse("!torch.vtensor")
        indices_type = Type.parse("!torch.vtensor")
        super(AtenTopkOp, self).__init__(values_type, indices_type, self_, k, dim, largest, sorted, loc=loc, ip=ip)
        
    
class AtenTransposeIntOp:
    def __init__(self, self_: "shark.Tensor", dim0: int, dim1: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim0):
            dim0 = torch_dialect.ConstantIntOp(dim0)
        else:
            dim0 = get_op_result_or_value(dim0)
            assert str(dim0.type) == '!torch.int', f'dim0 should a !torch.int but is {dim0.type}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert str(dim1.type) == '!torch.int', f'dim1 should a !torch.int but is {dim1.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTransposeIntOp, self).__init__(result_type, self_, dim0, dim1, loc=loc, ip=ip)
        
    
class AtenPermuteOp:
    def __init__(self, self_: "shark.Tensor", dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dims):
            dims = list(map(torch_dialect.ConstantIntOp, dims))
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert str(dims.type) == '!torch.list<int>', f'dims should a !torch.list<int> but is {dims.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenPermuteOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class AtenBmmOp:
    def __init__(self, self_: "shark.Tensor", mat2: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mat2):
            assert is_mlir_value(mat2), f'mat2 should be a Tensor but is {mat2.type}'
        else:
            mat2 = get_op_result_or_value(mat2)
            assert str(mat2.type).startswith("!torch.vtensor"), f'mat2 should a Tensor but is {mat2.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBmmOp, self).__init__(result_type, self_, mat2, loc=loc, ip=ip)
        
    
class AtenCumsumOp:
    def __init__(self, self_: "shark.Tensor", dim: int, dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCumsumOp, self).__init__(result_type, self_, dim, dtype, loc=loc, ip=ip)
        
    
class AtenFloorDivideScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFloorDivideScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenLogsumexpOp:
    def __init__(self, self_: "shark.Tensor", dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = list(map(torch_dialect.ConstantIntOp, dim))
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.list<int>', f'dim should a !torch.list<int> but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLogsumexpOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class Aten__And__TensorOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten__And__TensorOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class Aten_SoftmaxOp:
    def __init__(self, self_: "shark.Tensor", dim: int, half_to_float: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(half_to_float):
            half_to_float = torch_dialect.ConstantBoolOp(half_to_float)
        else:
            half_to_float = get_op_result_or_value(half_to_float)
            assert str(half_to_float.type) == '!torch.bool', f'half_to_float should a !torch.bool but is {half_to_float.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_SoftmaxOp, self).__init__(result_type, self_, dim, half_to_float, loc=loc, ip=ip)
        
    
class AtenMeanOp:
    def __init__(self, self_: "shark.Tensor", dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMeanOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenStdOp:
    def __init__(self, self_: "shark.Tensor", unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert str(unbiased.type) == '!torch.bool', f'unbiased should a !torch.bool but is {unbiased.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenStdOp, self).__init__(result_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenVarOp:
    def __init__(self, self_: "shark.Tensor", unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert str(unbiased.type) == '!torch.bool', f'unbiased should a !torch.bool but is {unbiased.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenVarOp, self).__init__(result_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenVarMeanOp:
    def __init__(self, self_: "shark.Tensor", unbiased: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(unbiased):
            unbiased = torch_dialect.ConstantBoolOp(unbiased)
        else:
            unbiased = get_op_result_or_value(unbiased)
            assert str(unbiased.type) == '!torch.bool', f'unbiased should a !torch.bool but is {unbiased.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        super(AtenVarMeanOp, self).__init__(result0_type, result1_type, self_, unbiased, loc=loc, ip=ip)
        
    
class AtenNllLossForwardOp:
    def __init__(self, self_: "shark.Tensor", target: "shark.Tensor", weight: Optional["shark.Tensor"], reduction: int, ignore_index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'target should be a Tensor but is {target.type}'
        else:
            target = get_op_result_or_value(target)
            assert str(target.type).startswith("!torch.vtensor"), f'target should a Tensor but is {target.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert str(reduction.type) == '!torch.int', f'reduction should a !torch.int but is {reduction.type}'
            
        if not is_mlir_value(ignore_index):
            ignore_index = torch_dialect.ConstantIntOp(ignore_index)
        else:
            ignore_index = get_op_result_or_value(ignore_index)
            assert str(ignore_index.type) == '!torch.int', f'ignore_index should a !torch.int but is {ignore_index.type}'
            
        output_type = Type.parse("!torch.vtensor")
        total_weight_type = Type.parse("!torch.vtensor")
        super(AtenNllLossForwardOp, self).__init__(output_type, total_weight_type, self_, target, weight, reduction, ignore_index, loc=loc, ip=ip)
        
    
class AtenNllLossBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", self_: "shark.Tensor", target: "shark.Tensor", weight: Optional["shark.Tensor"], reduction: int, ignore_index: int, total_weight: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'target should be a Tensor but is {target.type}'
        else:
            target = get_op_result_or_value(target)
            assert str(target.type).startswith("!torch.vtensor"), f'target should a Tensor but is {target.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert str(reduction.type) == '!torch.int', f'reduction should a !torch.int but is {reduction.type}'
            
        if not is_mlir_value(ignore_index):
            ignore_index = torch_dialect.ConstantIntOp(ignore_index)
        else:
            ignore_index = get_op_result_or_value(ignore_index)
            assert str(ignore_index.type) == '!torch.int', f'ignore_index should a !torch.int but is {ignore_index.type}'
            
        if not is_mlir_value(total_weight):
            assert is_mlir_value(total_weight), f'total_weight should be a Tensor but is {total_weight.type}'
        else:
            total_weight = get_op_result_or_value(total_weight)
            assert str(total_weight.type).startswith("!torch.vtensor"), f'total_weight should a Tensor but is {total_weight.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNllLossBackwardOp, self).__init__(result_type, grad_output, self_, target, weight, reduction, ignore_index, total_weight, loc=loc, ip=ip)
        
    
class AtenBincountOp:
    def __init__(self, self_: "shark.Tensor", weights: Optional["shark.Tensor"], minlength: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(weights):
            if weights is not None:
                assert is_mlir_value(weights), f'weights should be a Tensor but is {weights.type}'
            else:
                weights = torch_dialect.ConstantNoneOp()
        else:
            weights = get_op_result_or_value(weights)
            assert str(weights.type).startswith("!torch.vtensor"), f'weights should a Tensor but is {weights.type}'
            
        if not is_mlir_value(minlength):
            minlength = torch_dialect.ConstantIntOp(minlength)
        else:
            minlength = get_op_result_or_value(minlength)
            assert str(minlength.type) == '!torch.int', f'minlength should a !torch.int but is {minlength.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBincountOp, self).__init__(result_type, self_, weights, minlength, loc=loc, ip=ip)
        
    
class AtenFrobeniusNormDimOp:
    def __init__(self, self_: "shark.Tensor", dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = list(map(torch_dialect.ConstantIntOp, dim))
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.list<int>', f'dim should a !torch.list<int> but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFrobeniusNormDimOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenMseLossOp:
    def __init__(self, self_: "shark.Tensor", target: "shark.Tensor", reduction: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(target):
            assert is_mlir_value(target), f'target should be a Tensor but is {target.type}'
        else:
            target = get_op_result_or_value(target)
            assert str(target.type).startswith("!torch.vtensor"), f'target should a Tensor but is {target.type}'
            
        if not is_mlir_value(reduction):
            reduction = torch_dialect.ConstantIntOp(reduction)
        else:
            reduction = get_op_result_or_value(reduction)
            assert str(reduction.type) == '!torch.int', f'reduction should a !torch.int but is {reduction.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMseLossOp, self).__init__(result_type, self_, target, reduction, loc=loc, ip=ip)
        
    
class AtenUpsampleNearest2dBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", output_size: List[int], input_size: List[int], scales_h: Optional[float], scales_w: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(output_size):
            output_size = list(map(torch_dialect.ConstantIntOp, output_size))
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert str(output_size.type) == '!torch.list<int>', f'output_size should a !torch.list<int> but is {output_size.type}'
            
        if not is_mlir_value(input_size):
            input_size = list(map(torch_dialect.ConstantIntOp, input_size))
            input_size = torch_dialect.PrimListConstructOp(input_size)
        else:
            input_size = get_op_result_or_value(input_size)
            assert str(input_size.type) == '!torch.list<int>', f'input_size should a !torch.list<int> but is {input_size.type}'
            
        if not is_mlir_value(scales_h):
            if scales_h is not None:
                scales_h = torch_dialect.ConstantFloatOp(scales_h)
            else:
                scales_h = torch_dialect.ConstantNoneOp()
        else:
            scales_h = get_op_result_or_value(scales_h)
            assert str(scales_h.type) == '!torch.float', f'scales_h should a !torch.float but is {scales_h.type}'
            
        if not is_mlir_value(scales_w):
            if scales_w is not None:
                scales_w = torch_dialect.ConstantFloatOp(scales_w)
            else:
                scales_w = torch_dialect.ConstantNoneOp()
        else:
            scales_w = get_op_result_or_value(scales_w)
            assert str(scales_w.type) == '!torch.float', f'scales_w should a !torch.float but is {scales_w.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUpsampleNearest2dBackwardOp, self).__init__(result_type, grad_output, output_size, input_size, scales_h, scales_w, loc=loc, ip=ip)
        
    
class AtenConstantPadNdOp:
    def __init__(self, self_: "shark.Tensor", pad: List[int], value: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(pad):
            pad = list(map(torch_dialect.ConstantIntOp, pad))
            pad = torch_dialect.PrimListConstructOp(pad)
        else:
            pad = get_op_result_or_value(pad)
            assert str(pad.type) == '!torch.list<int>', f'pad should a !torch.list<int> but is {pad.type}'
            
        if not is_mlir_value(value):
            value = torch_dialect.ConstantNumberOp(value)
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) in {'!torch.float', '!torch.int'}, f'value should a !torch.number but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenConstantPadNdOp, self).__init__(result_type, self_, pad, value, loc=loc, ip=ip)
        
    
class AtenPadOp:
    def __init__(self, self_: "shark.Tensor", pad: List[int], mode: str, value: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(pad):
            pad = list(map(torch_dialect.ConstantIntOp, pad))
            pad = torch_dialect.PrimListConstructOp(pad)
        else:
            pad = get_op_result_or_value(pad)
            assert str(pad.type) == '!torch.list<int>', f'pad should a !torch.list<int> but is {pad.type}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantStrOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert str(mode.type) == '!torch.str', f'mode should a !torch.str but is {mode.type}'
            
        if not is_mlir_value(value):
            if value is not None:
                value = torch_dialect.ConstantFloatOp(value)
            else:
                value = torch_dialect.ConstantNoneOp()
        else:
            value = get_op_result_or_value(value)
            assert str(value.type) == '!torch.float', f'value should a !torch.float but is {value.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenPadOp, self).__init__(result_type, self_, pad, mode, value, loc=loc, ip=ip)
        
    
class AtenSqueezeDimOp:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqueezeDimOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenSqueezeOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqueezeOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenFlattenUsingIntsOp:
    def __init__(self, self_: "shark.Tensor", start_dim: int, end_dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(start_dim):
            start_dim = torch_dialect.ConstantIntOp(start_dim)
        else:
            start_dim = get_op_result_or_value(start_dim)
            assert str(start_dim.type) == '!torch.int', f'start_dim should a !torch.int but is {start_dim.type}'
            
        if not is_mlir_value(end_dim):
            end_dim = torch_dialect.ConstantIntOp(end_dim)
        else:
            end_dim = get_op_result_or_value(end_dim)
            assert str(end_dim.type) == '!torch.int', f'end_dim should a !torch.int but is {end_dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFlattenUsingIntsOp, self).__init__(result_type, self_, start_dim, end_dim, loc=loc, ip=ip)
        
    
class AtenDimOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        super(AtenDimOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenSizeOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.list<int>")
        super(AtenSizeOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBoolTensorOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenBoolTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIsFloatingPointOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        super(AtenIsFloatingPointOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class Aten_ShapeAsTensorOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_ShapeAsTensorOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAllOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAllOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAllBoolOp:
    def __init__(self, self_: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = list(map(torch_dialect.ConstantBoolOp, self_))
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            # should be bool[]
            pass
            
        super(AtenAllBoolOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenAnyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAnyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAnyDimOp:
    def __init__(self, self_: "shark.Tensor", dim: int, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAnyDimOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenArangeStartOutOp:
    def __init__(self, start: "shark.Number", end: "shark.Number", step: "shark.Number", out: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(start):
            start = torch_dialect.ConstantNumberOp(start)
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) in {'!torch.float', '!torch.int'}, f'start should a !torch.number but is {start.type}'
            
        if not is_mlir_value(end):
            end = torch_dialect.ConstantNumberOp(end)
        else:
            end = get_op_result_or_value(end)
            assert str(end.type) in {'!torch.float', '!torch.int'}, f'end should a !torch.number but is {end.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantNumberOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) in {'!torch.float', '!torch.int'}, f'step should a !torch.number but is {step.type}'
            
        if not is_mlir_value(out):
            assert is_mlir_value(out), f'out should be a Tensor but is {out.type}'
        else:
            out = get_op_result_or_value(out)
            assert str(out.type).startswith("!torch.vtensor"), f'out should a Tensor but is {out.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenArangeStartOutOp, self).__init__(result_type, start, end, step, out, loc=loc, ip=ip)
        
    
class AtenArgmaxOp:
    def __init__(self, self_: "shark.Tensor", dim: Optional[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            if dim is not None:
                dim = torch_dialect.ConstantIntOp(dim)
            else:
                dim = torch_dialect.ConstantNoneOp()
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenArgmaxOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenBucketizeTensorOp:
    def __init__(self, self_: "shark.Tensor", boundaries: "shark.Tensor", out_int32: bool, right: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(boundaries):
            assert is_mlir_value(boundaries), f'boundaries should be a Tensor but is {boundaries.type}'
        else:
            boundaries = get_op_result_or_value(boundaries)
            assert str(boundaries.type).startswith("!torch.vtensor"), f'boundaries should a Tensor but is {boundaries.type}'
            
        if not is_mlir_value(out_int32):
            out_int32 = torch_dialect.ConstantBoolOp(out_int32)
        else:
            out_int32 = get_op_result_or_value(out_int32)
            assert str(out_int32.type) == '!torch.bool', f'out_int32 should a !torch.bool but is {out_int32.type}'
            
        if not is_mlir_value(right):
            right = torch_dialect.ConstantBoolOp(right)
        else:
            right = get_op_result_or_value(right)
            assert str(right.type) == '!torch.bool', f'right should a !torch.bool but is {right.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBucketizeTensorOp, self).__init__(result_type, self_, boundaries, out_int32, right, loc=loc, ip=ip)
        
    
class AtenCloneOp:
    def __init__(self, self_: "shark.Tensor", memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert str(memory_format.type) == '!torch.int', f'memory_format should a !torch.int but is {memory_format.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCloneOp, self).__init__(result_type, self_, memory_format, loc=loc, ip=ip)
        
    
class AtenLiftFreshCopyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenLiftFreshCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenContiguousOp:
    def __init__(self, self_: "shark.Tensor", memory_format: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(memory_format):
            memory_format = torch_dialect.ConstantIntOp(memory_format)
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert str(memory_format.type) == '!torch.int', f'memory_format should a !torch.int but is {memory_format.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenContiguousOp, self).__init__(result_type, self_, memory_format, loc=loc, ip=ip)
        
    
class AtenCopyOp:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", non_blocking: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert str(non_blocking.type) == '!torch.bool', f'non_blocking should a !torch.bool but is {non_blocking.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCopyOp, self).__init__(result_type, self_, src, non_blocking, loc=loc, ip=ip)
        
    
class AtenCopy_Op:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", non_blocking: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert str(non_blocking.type) == '!torch.bool', f'non_blocking should a !torch.bool but is {non_blocking.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCopy_Op, self).__init__(result_type, self_, src, non_blocking, loc=loc, ip=ip)
        
    
class AtenDetachOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDetachOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenEmbeddingOp:
    def __init__(self, weight: "shark.Tensor", indices: "shark.Tensor", padding_idx: int, scale_grad_by_freq: bool, sparse: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'indices should be a Tensor but is {indices.type}'
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type).startswith("!torch.vtensor"), f'indices should a Tensor but is {indices.type}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert str(padding_idx.type) == '!torch.int', f'padding_idx should a !torch.int but is {padding_idx.type}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert str(scale_grad_by_freq.type) == '!torch.bool', f'scale_grad_by_freq should a !torch.bool but is {scale_grad_by_freq.type}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert str(sparse.type) == '!torch.bool', f'sparse should a !torch.bool but is {sparse.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEmbeddingOp, self).__init__(result_type, weight, indices, padding_idx, scale_grad_by_freq, sparse, loc=loc, ip=ip)
        
    
class AtenEmbeddingBagPaddingIdxOp:
    def __init__(self, weight: "shark.Tensor", indices: "shark.Tensor", offsets: "shark.Tensor", scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional["shark.Tensor"], include_last_offset: bool, padding_idx: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'indices should be a Tensor but is {indices.type}'
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type).startswith("!torch.vtensor"), f'indices should a Tensor but is {indices.type}'
            
        if not is_mlir_value(offsets):
            assert is_mlir_value(offsets), f'offsets should be a Tensor but is {offsets.type}'
        else:
            offsets = get_op_result_or_value(offsets)
            assert str(offsets.type).startswith("!torch.vtensor"), f'offsets should a Tensor but is {offsets.type}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert str(scale_grad_by_freq.type) == '!torch.bool', f'scale_grad_by_freq should a !torch.bool but is {scale_grad_by_freq.type}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantIntOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert str(mode.type) == '!torch.int', f'mode should a !torch.int but is {mode.type}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert str(sparse.type) == '!torch.bool', f'sparse should a !torch.bool but is {sparse.type}'
            
        if not is_mlir_value(per_sample_weights):
            if per_sample_weights is not None:
                assert is_mlir_value(per_sample_weights), f'per_sample_weights should be a Tensor but is {per_sample_weights.type}'
            else:
                per_sample_weights = torch_dialect.ConstantNoneOp()
        else:
            per_sample_weights = get_op_result_or_value(per_sample_weights)
            assert str(per_sample_weights.type).startswith("!torch.vtensor"), f'per_sample_weights should a Tensor but is {per_sample_weights.type}'
            
        if not is_mlir_value(include_last_offset):
            include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset)
        else:
            include_last_offset = get_op_result_or_value(include_last_offset)
            assert str(include_last_offset.type) == '!torch.bool', f'include_last_offset should a !torch.bool but is {include_last_offset.type}'
            
        if not is_mlir_value(padding_idx):
            if padding_idx is not None:
                padding_idx = torch_dialect.ConstantIntOp(padding_idx)
            else:
                padding_idx = torch_dialect.ConstantNoneOp()
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert str(padding_idx.type) == '!torch.int', f'padding_idx should a !torch.int but is {padding_idx.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        result3_type = Type.parse("!torch.vtensor")
        super(AtenEmbeddingBagPaddingIdxOp, self).__init__(result0_type, result1_type, result2_type, result3_type, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc=loc, ip=ip)
        
    
class Aten_EmbeddingBagOp:
    def __init__(self, weight: "shark.Tensor", indices: "shark.Tensor", offsets: "shark.Tensor", scale_grad_by_freq: bool, mode: int, sparse: bool, per_sample_weights: Optional["shark.Tensor"], include_last_offset: bool, padding_idx: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(weight):
            assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'indices should be a Tensor but is {indices.type}'
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type).startswith("!torch.vtensor"), f'indices should a Tensor but is {indices.type}'
            
        if not is_mlir_value(offsets):
            assert is_mlir_value(offsets), f'offsets should be a Tensor but is {offsets.type}'
        else:
            offsets = get_op_result_or_value(offsets)
            assert str(offsets.type).startswith("!torch.vtensor"), f'offsets should a Tensor but is {offsets.type}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert str(scale_grad_by_freq.type) == '!torch.bool', f'scale_grad_by_freq should a !torch.bool but is {scale_grad_by_freq.type}'
            
        if not is_mlir_value(mode):
            mode = torch_dialect.ConstantIntOp(mode)
        else:
            mode = get_op_result_or_value(mode)
            assert str(mode.type) == '!torch.int', f'mode should a !torch.int but is {mode.type}'
            
        if not is_mlir_value(sparse):
            sparse = torch_dialect.ConstantBoolOp(sparse)
        else:
            sparse = get_op_result_or_value(sparse)
            assert str(sparse.type) == '!torch.bool', f'sparse should a !torch.bool but is {sparse.type}'
            
        if not is_mlir_value(per_sample_weights):
            if per_sample_weights is not None:
                assert is_mlir_value(per_sample_weights), f'per_sample_weights should be a Tensor but is {per_sample_weights.type}'
            else:
                per_sample_weights = torch_dialect.ConstantNoneOp()
        else:
            per_sample_weights = get_op_result_or_value(per_sample_weights)
            assert str(per_sample_weights.type).startswith("!torch.vtensor"), f'per_sample_weights should a Tensor but is {per_sample_weights.type}'
            
        if not is_mlir_value(include_last_offset):
            include_last_offset = torch_dialect.ConstantBoolOp(include_last_offset)
        else:
            include_last_offset = get_op_result_or_value(include_last_offset)
            assert str(include_last_offset.type) == '!torch.bool', f'include_last_offset should a !torch.bool but is {include_last_offset.type}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert str(padding_idx.type) == '!torch.int', f'padding_idx should a !torch.int but is {padding_idx.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        result3_type = Type.parse("!torch.vtensor")
        super(Aten_EmbeddingBagOp, self).__init__(result0_type, result1_type, result2_type, result3_type, weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx, loc=loc, ip=ip)
        
    
class AtenExpandOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], implicit: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(implicit):
            implicit = torch_dialect.ConstantBoolOp(implicit)
        else:
            implicit = get_op_result_or_value(implicit)
            assert str(implicit.type) == '!torch.bool', f'implicit should a !torch.bool but is {implicit.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpandOp, self).__init__(result_type, self_, size, implicit, loc=loc, ip=ip)
        
    
class AtenExpandAsOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpandAsOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenBroadcastToOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBroadcastToOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenIndexTensorHackedTwinOp:
    def __init__(self, self_: "shark.Tensor", indices: List["shark.Tensor"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(indices):
            indices = torch_dialect.PrimListConstructOp(indices)
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type) == '!torch.list<Tensor>', f'indices should a !torch.list<Tensor> but is {indices.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenIndexTensorHackedTwinOp, self).__init__(result_type, self_, indices, loc=loc, ip=ip)
        
    
class AtenIndexSelectOp:
    def __init__(self, self_: "shark.Tensor", dim: int, index: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'index should be a Tensor but is {index.type}'
        else:
            index = get_op_result_or_value(index)
            assert str(index.type).startswith("!torch.vtensor"), f'index should a Tensor but is {index.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenIndexSelectOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class AtenItemOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        super(AtenItemOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenMaskedSelectOp:
    def __init__(self, self_: "shark.Tensor", mask: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaskedSelectOp, self).__init__(result_type, self_, mask, loc=loc, ip=ip)
        
    
class AtenNumelOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        super(AtenNumelOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenRepeatOp:
    def __init__(self, self_: "shark.Tensor", repeats: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(repeats):
            repeats = list(map(torch_dialect.ConstantIntOp, repeats))
            repeats = torch_dialect.PrimListConstructOp(repeats)
        else:
            repeats = get_op_result_or_value(repeats)
            assert str(repeats.type) == '!torch.list<int>', f'repeats should a !torch.list<int> but is {repeats.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRepeatOp, self).__init__(result_type, self_, repeats, loc=loc, ip=ip)
        
    
class AtenReshapeOp:
    def __init__(self, self_: "shark.Tensor", shape: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(shape):
            shape = list(map(torch_dialect.ConstantIntOp, shape))
            shape = torch_dialect.PrimListConstructOp(shape)
        else:
            shape = get_op_result_or_value(shape)
            assert str(shape.type) == '!torch.list<int>', f'shape should a !torch.list<int> but is {shape.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenReshapeOp, self).__init__(result_type, self_, shape, loc=loc, ip=ip)
        
    
class Aten_ReshapeAliasOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], stride: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_ReshapeAliasOp, self).__init__(result_type, self_, size, stride, loc=loc, ip=ip)
        
    
class AtenResize_Op:
    def __init__(self, self_: "shark.Tensor", size: List[int], memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert str(memory_format.type) == '!torch.int', f'memory_format should a !torch.int but is {memory_format.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenResize_Op, self).__init__(result_type, self_, size, memory_format, loc=loc, ip=ip)
        
    
class AtenSelectIntOp:
    def __init__(self, self_: "shark.Tensor", dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert str(index.type) == '!torch.int', f'index should a !torch.int but is {index.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSelectIntOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class AtenSizeIntOp:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        super(AtenSizeIntOp, self).__init__(self_, dim, loc=loc, ip=ip)
        
    
class AtenStackOp:
    def __init__(self, tensors: List["shark.Tensor"], dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tensors):
            tensors = torch_dialect.PrimListConstructOp(tensors)
        else:
            tensors = get_op_result_or_value(tensors)
            assert str(tensors.type) == '!torch.list<Tensor>', f'tensors should a !torch.list<Tensor> but is {tensors.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenStackOp, self).__init__(result_type, tensors, dim, loc=loc, ip=ip)
        
    
class AtenSumOp:
    def __init__(self, self_: "shark.Tensor", dtype: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dtype):
            if dtype is not None:
                dtype = torch_dialect.ConstantIntOp(dtype)
            else:
                dtype = torch_dialect.ConstantNoneOp()
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSumOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenMaxOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenMaxOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenMaxDimOp:
    def __init__(self, self_: "shark.Tensor", dim: int, keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        values_type = Type.parse("!torch.vtensor")
        indices_type = Type.parse("!torch.vtensor")
        super(AtenMaxDimOp, self).__init__(values_type, indices_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenAmaxOp:
    def __init__(self, self_: "shark.Tensor", dim: List[int], keepdim: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = list(map(torch_dialect.ConstantIntOp, dim))
            dim = torch_dialect.PrimListConstructOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.list<int>', f'dim should a !torch.list<int> but is {dim.type}'
            
        if not is_mlir_value(keepdim):
            keepdim = torch_dialect.ConstantBoolOp(keepdim)
        else:
            keepdim = get_op_result_or_value(keepdim)
            assert str(keepdim.type) == '!torch.bool', f'keepdim should a !torch.bool but is {keepdim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAmaxOp, self).__init__(result_type, self_, dim, keepdim, loc=loc, ip=ip)
        
    
class AtenToDtypeOp:
    def __init__(self, self_: "shark.Tensor", dtype: int, non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert str(non_blocking.type) == '!torch.bool', f'non_blocking should a !torch.bool but is {non_blocking.type}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert str(copy.type) == '!torch.bool', f'copy should a !torch.bool but is {copy.type}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert str(memory_format.type) == '!torch.int', f'memory_format should a !torch.int but is {memory_format.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenToDtypeOp, self).__init__(result_type, self_, dtype, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenToOtherOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", non_blocking: bool, copy: bool, memory_format: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        if not is_mlir_value(non_blocking):
            non_blocking = torch_dialect.ConstantBoolOp(non_blocking)
        else:
            non_blocking = get_op_result_or_value(non_blocking)
            assert str(non_blocking.type) == '!torch.bool', f'non_blocking should a !torch.bool but is {non_blocking.type}'
            
        if not is_mlir_value(copy):
            copy = torch_dialect.ConstantBoolOp(copy)
        else:
            copy = get_op_result_or_value(copy)
            assert str(copy.type) == '!torch.bool', f'copy should a !torch.bool but is {copy.type}'
            
        if not is_mlir_value(memory_format):
            if memory_format is not None:
                memory_format = torch_dialect.ConstantIntOp(memory_format)
            else:
                memory_format = torch_dialect.ConstantNoneOp()
        else:
            memory_format = get_op_result_or_value(memory_format)
            assert str(memory_format.type) == '!torch.int', f'memory_format should a !torch.int but is {memory_format.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenToOtherOp, self).__init__(result_type, self_, other, non_blocking, copy, memory_format, loc=loc, ip=ip)
        
    
class AtenTypeAsOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTypeAsOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenViewOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenViewOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class Aten_UnsafeViewOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_UnsafeViewOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenWhereSelfOp:
    def __init__(self, condition: "shark.Tensor", self_: "shark.Tensor", other: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'condition should be a Tensor but is {condition.type}'
        else:
            condition = get_op_result_or_value(condition)
            assert str(condition.type).startswith("!torch.vtensor"), f'condition should a Tensor but is {condition.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenWhereSelfOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarOp:
    def __init__(self, condition: "shark.Tensor", self_: "shark.Number", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'condition should be a Tensor but is {condition.type}'
        else:
            condition = get_op_result_or_value(condition)
            assert str(condition.type).startswith("!torch.vtensor"), f'condition should a Tensor but is {condition.type}'
            
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantNumberOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) in {'!torch.float', '!torch.int'}, f'self_ should a !torch.number but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenWhereScalarOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarOtherOp:
    def __init__(self, condition: "shark.Tensor", self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'condition should be a Tensor but is {condition.type}'
        else:
            condition = get_op_result_or_value(condition)
            assert str(condition.type).startswith("!torch.vtensor"), f'condition should a Tensor but is {condition.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenWhereScalarOtherOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenWhereScalarSelfOp:
    def __init__(self, condition: "shark.Tensor", self_: "shark.Number", other: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(condition):
            assert is_mlir_value(condition), f'condition should be a Tensor but is {condition.type}'
        else:
            condition = get_op_result_or_value(condition)
            assert str(condition.type).startswith("!torch.vtensor"), f'condition should a Tensor but is {condition.type}'
            
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantNumberOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) in {'!torch.float', '!torch.int'}, f'self_ should a !torch.number but is {self_.type}'
            
        if not is_mlir_value(other):
            assert is_mlir_value(other), f'other should be a Tensor but is {other.type}'
        else:
            other = get_op_result_or_value(other)
            assert str(other.type).startswith("!torch.vtensor"), f'other should a Tensor but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenWhereScalarSelfOp, self).__init__(result_type, condition, self_, other, loc=loc, ip=ip)
        
    
class AtenSliceTensorOp:
    def __init__(self, self_: "shark.Tensor", dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert str(end.type) == '!torch.int', f'end should a !torch.int but is {end.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSliceTensorOp, self).__init__(result_type, self_, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenLenTensorOp:
    def __init__(self, t: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(t):
            assert is_mlir_value(t), f't should be a Tensor but is {t.type}'
        else:
            t = get_op_result_or_value(t)
            assert str(t.type).startswith("!torch.vtensor"), f't should a Tensor but is {t.type}'
            
        super(AtenLenTensorOp, self).__init__(t, loc=loc, ip=ip)
        
    
class AtenCpuOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCpuOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenGatherOp:
    def __init__(self, self_: "shark.Tensor", dim: int, index: "shark.Tensor", sparse_grad: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'index should be a Tensor but is {index.type}'
        else:
            index = get_op_result_or_value(index)
            assert str(index.type).startswith("!torch.vtensor"), f'index should a Tensor but is {index.type}'
            
        if not is_mlir_value(sparse_grad):
            sparse_grad = torch_dialect.ConstantBoolOp(sparse_grad)
        else:
            sparse_grad = get_op_result_or_value(sparse_grad)
            assert str(sparse_grad.type) == '!torch.bool', f'sparse_grad should a !torch.bool but is {sparse_grad.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGatherOp, self).__init__(result_type, self_, dim, index, sparse_grad, loc=loc, ip=ip)
        
    
class AtenScatterAddOp:
    def __init__(self, self_: "shark.Tensor", dim: int, index: "shark.Tensor", src: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            assert is_mlir_value(index), f'index should be a Tensor but is {index.type}'
        else:
            index = get_op_result_or_value(index)
            assert str(index.type).startswith("!torch.vtensor"), f'index should a Tensor but is {index.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenScatterAddOp, self).__init__(result_type, self_, dim, index, src, loc=loc, ip=ip)
        
    
class AtenIntImplicitOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenIntImplicitOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatImplicitOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenFloatImplicitOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIntTensorOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenIntTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatTensorOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenFloatTensorOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenDropoutOp:
    def __init__(self, input: "shark.Tensor", p: float, train: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert str(p.type) == '!torch.float', f'p should a !torch.float but is {p.type}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert str(train.type) == '!torch.bool', f'train should a !torch.bool but is {train.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDropoutOp, self).__init__(result_type, input, p, train, loc=loc, ip=ip)
        
    
class AtenDropout_Op:
    def __init__(self, self_: "shark.Tensor", p: float, train: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert str(p.type) == '!torch.float', f'p should a !torch.float but is {p.type}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert str(train.type) == '!torch.bool', f'train should a !torch.bool but is {train.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDropout_Op, self).__init__(result_type, self_, p, train, loc=loc, ip=ip)
        
    
class AtenNativeDropoutOp:
    def __init__(self, input: "shark.Tensor", p: float, train: Optional[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(p):
            p = torch_dialect.ConstantFloatOp(p)
        else:
            p = get_op_result_or_value(p)
            assert str(p.type) == '!torch.float', f'p should a !torch.float but is {p.type}'
            
        if not is_mlir_value(train):
            if train is not None:
                train = torch_dialect.ConstantBoolOp(train)
            else:
                train = torch_dialect.ConstantNoneOp()
        else:
            train = get_op_result_or_value(train)
            assert str(train.type) == '!torch.bool', f'train should a !torch.bool but is {train.type}'
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        super(AtenNativeDropoutOp, self).__init__(result0_type, result1_type, input, p, train, loc=loc, ip=ip)
        
    
class AtenTOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenNumpyTOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNumpyTOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenBaddbmmOp:
    def __init__(self, self_: "shark.Tensor", batch1: "shark.Tensor", batch2: "shark.Tensor", beta: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(batch1):
            assert is_mlir_value(batch1), f'batch1 should be a Tensor but is {batch1.type}'
        else:
            batch1 = get_op_result_or_value(batch1)
            assert str(batch1.type).startswith("!torch.vtensor"), f'batch1 should a Tensor but is {batch1.type}'
            
        if not is_mlir_value(batch2):
            assert is_mlir_value(batch2), f'batch2 should be a Tensor but is {batch2.type}'
        else:
            batch2 = get_op_result_or_value(batch2)
            assert str(batch2.type).startswith("!torch.vtensor"), f'batch2 should a Tensor but is {batch2.type}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert str(beta.type) in {'!torch.float', '!torch.int'}, f'beta should a !torch.number but is {beta.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBaddbmmOp, self).__init__(result_type, self_, batch1, batch2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenBaddbmm_Op:
    def __init__(self, self_: "shark.Tensor", batch1: "shark.Tensor", batch2: "shark.Tensor", beta: "shark.Number", alpha: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(batch1):
            assert is_mlir_value(batch1), f'batch1 should be a Tensor but is {batch1.type}'
        else:
            batch1 = get_op_result_or_value(batch1)
            assert str(batch1.type).startswith("!torch.vtensor"), f'batch1 should a Tensor but is {batch1.type}'
            
        if not is_mlir_value(batch2):
            assert is_mlir_value(batch2), f'batch2 should be a Tensor but is {batch2.type}'
        else:
            batch2 = get_op_result_or_value(batch2)
            assert str(batch2.type).startswith("!torch.vtensor"), f'batch2 should a Tensor but is {batch2.type}'
            
        if not is_mlir_value(beta):
            beta = torch_dialect.ConstantNumberOp(beta)
        else:
            beta = get_op_result_or_value(beta)
            assert str(beta.type) in {'!torch.float', '!torch.int'}, f'beta should a !torch.number but is {beta.type}'
            
        if not is_mlir_value(alpha):
            alpha = torch_dialect.ConstantNumberOp(alpha)
        else:
            alpha = get_op_result_or_value(alpha)
            assert str(alpha.type) in {'!torch.float', '!torch.int'}, f'alpha should a !torch.number but is {alpha.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenBaddbmm_Op, self).__init__(result_type, self_, batch1, batch2, beta, alpha, loc=loc, ip=ip)
        
    
class AtenFftFftOp:
    def __init__(self, self_: "shark.Tensor", n: Optional[int], dim: int, norm: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(n):
            if n is not None:
                n = torch_dialect.ConstantIntOp(n)
            else:
                n = torch_dialect.ConstantNoneOp()
        else:
            n = get_op_result_or_value(n)
            assert str(n.type) == '!torch.int', f'n should a !torch.int but is {n.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(norm):
            if norm is not None:
                norm = torch_dialect.ConstantStrOp(norm)
            else:
                norm = torch_dialect.ConstantNoneOp()
        else:
            norm = get_op_result_or_value(norm)
            assert str(norm.type) == '!torch.str', f'norm should a !torch.str but is {norm.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenFftFftOp, self).__init__(result_type, self_, n, dim, norm, loc=loc, ip=ip)
        
    
class AtenAliasCopyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAliasCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenAsStridedCopyOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], stride: List[int], storage_offset: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(storage_offset):
            if storage_offset is not None:
                storage_offset = torch_dialect.ConstantIntOp(storage_offset)
            else:
                storage_offset = torch_dialect.ConstantNoneOp()
        else:
            storage_offset = get_op_result_or_value(storage_offset)
            assert str(storage_offset.type) == '!torch.int', f'storage_offset should a !torch.int but is {storage_offset.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAsStridedCopyOp, self).__init__(result_type, self_, size, stride, storage_offset, loc=loc, ip=ip)
        
    
class AtenDiagonalCopyOp:
    def __init__(self, self_: "shark.Tensor", offset: int, dim1: int, dim2: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(offset):
            offset = torch_dialect.ConstantIntOp(offset)
        else:
            offset = get_op_result_or_value(offset)
            assert str(offset.type) == '!torch.int', f'offset should a !torch.int but is {offset.type}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert str(dim1.type) == '!torch.int', f'dim1 should a !torch.int but is {dim1.type}'
            
        if not is_mlir_value(dim2):
            dim2 = torch_dialect.ConstantIntOp(dim2)
        else:
            dim2 = get_op_result_or_value(dim2)
            assert str(dim2.type) == '!torch.int', f'dim2 should a !torch.int but is {dim2.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDiagonalCopyOp, self).__init__(result_type, self_, offset, dim1, dim2, loc=loc, ip=ip)
        
    
class AtenExpandCopyOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], implicit: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(implicit):
            implicit = torch_dialect.ConstantBoolOp(implicit)
        else:
            implicit = get_op_result_or_value(implicit)
            assert str(implicit.type) == '!torch.bool', f'implicit should a !torch.bool but is {implicit.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenExpandCopyOp, self).__init__(result_type, self_, size, implicit, loc=loc, ip=ip)
        
    
class AtenPermuteCopyOp:
    def __init__(self, self_: "shark.Tensor", dims: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dims):
            dims = list(map(torch_dialect.ConstantIntOp, dims))
            dims = torch_dialect.PrimListConstructOp(dims)
        else:
            dims = get_op_result_or_value(dims)
            assert str(dims.type) == '!torch.list<int>', f'dims should a !torch.list<int> but is {dims.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenPermuteCopyOp, self).__init__(result_type, self_, dims, loc=loc, ip=ip)
        
    
class Aten_ReshapeAliasCopyOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], stride: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_ReshapeAliasCopyOp, self).__init__(result_type, self_, size, stride, loc=loc, ip=ip)
        
    
class AtenSelectCopyIntOp:
    def __init__(self, self_: "shark.Tensor", dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert str(index.type) == '!torch.int', f'index should a !torch.int but is {index.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSelectCopyIntOp, self).__init__(result_type, self_, dim, index, loc=loc, ip=ip)
        
    
class AtenDetachCopyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDetachCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSliceCopyTensorOp:
    def __init__(self, self_: "shark.Tensor", dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert str(end.type) == '!torch.int', f'end should a !torch.int but is {end.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSliceCopyTensorOp, self).__init__(result_type, self_, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenSqueezeCopyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqueezeCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenSqueezeCopyDimOp:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSqueezeCopyDimOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenTCopyOp:
    def __init__(self, self_: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTCopyOp, self).__init__(result_type, self_, loc=loc, ip=ip)
        
    
class AtenTransposeCopyIntOp:
    def __init__(self, self_: "shark.Tensor", dim0: int, dim1: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim0):
            dim0 = torch_dialect.ConstantIntOp(dim0)
        else:
            dim0 = get_op_result_or_value(dim0)
            assert str(dim0.type) == '!torch.int', f'dim0 should a !torch.int but is {dim0.type}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert str(dim1.type) == '!torch.int', f'dim1 should a !torch.int but is {dim1.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTransposeCopyIntOp, self).__init__(result_type, self_, dim0, dim1, loc=loc, ip=ip)
        
    
class AtenUnsqueezeCopyOp:
    def __init__(self, self_: "shark.Tensor", dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUnsqueezeCopyOp, self).__init__(result_type, self_, dim, loc=loc, ip=ip)
        
    
class AtenViewCopyOp:
    def __init__(self, self_: "shark.Tensor", size: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenViewCopyOp, self).__init__(result_type, self_, size, loc=loc, ip=ip)
        
    
class AtenViewCopyDtypeOp:
    def __init__(self, self_: "shark.Tensor", dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenViewCopyDtypeOp, self).__init__(result_type, self_, dtype, loc=loc, ip=ip)
        
    
class AtenUnfoldCopyOp:
    def __init__(self, self_: "shark.Tensor", dimension: int, size: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dimension):
            dimension = torch_dialect.ConstantIntOp(dimension)
        else:
            dimension = get_op_result_or_value(dimension)
            assert str(dimension.type) == '!torch.int', f'dimension should a !torch.int but is {dimension.type}'
            
        if not is_mlir_value(size):
            size = torch_dialect.ConstantIntOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.int', f'size should a !torch.int but is {size.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUnfoldCopyOp, self).__init__(result_type, self_, dimension, size, step, loc=loc, ip=ip)
        
    
class AtenSelectScatterOp:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", dim: int, index: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert str(index.type) == '!torch.int', f'index should a !torch.int but is {index.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSelectScatterOp, self).__init__(result_type, self_, src, dim, index, loc=loc, ip=ip)
        
    
class AtenSliceScatterOp:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", dim: int, start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert str(end.type) == '!torch.int', f'end should a !torch.int but is {end.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenSliceScatterOp, self).__init__(result_type, self_, src, dim, start, end, step, loc=loc, ip=ip)
        
    
class AtenDiagonalScatterOp:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", offset: int, dim1: int, dim2: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(offset):
            offset = torch_dialect.ConstantIntOp(offset)
        else:
            offset = get_op_result_or_value(offset)
            assert str(offset.type) == '!torch.int', f'offset should a !torch.int but is {offset.type}'
            
        if not is_mlir_value(dim1):
            dim1 = torch_dialect.ConstantIntOp(dim1)
        else:
            dim1 = get_op_result_or_value(dim1)
            assert str(dim1.type) == '!torch.int', f'dim1 should a !torch.int but is {dim1.type}'
            
        if not is_mlir_value(dim2):
            dim2 = torch_dialect.ConstantIntOp(dim2)
        else:
            dim2 = get_op_result_or_value(dim2)
            assert str(dim2.type) == '!torch.int', f'dim2 should a !torch.int but is {dim2.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenDiagonalScatterOp, self).__init__(result_type, self_, src, offset, dim1, dim2, loc=loc, ip=ip)
        
    
class AtenAsStridedScatterOp:
    def __init__(self, self_: "shark.Tensor", src: "shark.Tensor", size: List[int], stride: List[int], storage_offset: Optional[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(src):
            assert is_mlir_value(src), f'src should be a Tensor but is {src.type}'
        else:
            src = get_op_result_or_value(src)
            assert str(src.type).startswith("!torch.vtensor"), f'src should a Tensor but is {src.type}'
            
        if not is_mlir_value(size):
            size = list(map(torch_dialect.ConstantIntOp, size))
            size = torch_dialect.PrimListConstructOp(size)
        else:
            size = get_op_result_or_value(size)
            assert str(size.type) == '!torch.list<int>', f'size should a !torch.list<int> but is {size.type}'
            
        if not is_mlir_value(stride):
            stride = list(map(torch_dialect.ConstantIntOp, stride))
            stride = torch_dialect.PrimListConstructOp(stride)
        else:
            stride = get_op_result_or_value(stride)
            assert str(stride.type) == '!torch.list<int>', f'stride should a !torch.list<int> but is {stride.type}'
            
        if not is_mlir_value(storage_offset):
            if storage_offset is not None:
                storage_offset = torch_dialect.ConstantIntOp(storage_offset)
            else:
                storage_offset = torch_dialect.ConstantNoneOp()
        else:
            storage_offset = get_op_result_or_value(storage_offset)
            assert str(storage_offset.type) == '!torch.int', f'storage_offset should a !torch.int but is {storage_offset.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenAsStridedScatterOp, self).__init__(result_type, self_, src, size, stride, storage_offset, loc=loc, ip=ip)
        
    
class AtenUpsampleNearest2dOp:
    def __init__(self, self_: "shark.Tensor", output_size: List[int], scales_h: Optional[float], scales_w: Optional[float], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(output_size):
            output_size = list(map(torch_dialect.ConstantIntOp, output_size))
            output_size = torch_dialect.PrimListConstructOp(output_size)
        else:
            output_size = get_op_result_or_value(output_size)
            assert str(output_size.type) == '!torch.list<int>', f'output_size should a !torch.list<int> but is {output_size.type}'
            
        if not is_mlir_value(scales_h):
            if scales_h is not None:
                scales_h = torch_dialect.ConstantFloatOp(scales_h)
            else:
                scales_h = torch_dialect.ConstantNoneOp()
        else:
            scales_h = get_op_result_or_value(scales_h)
            assert str(scales_h.type) == '!torch.float', f'scales_h should a !torch.float but is {scales_h.type}'
            
        if not is_mlir_value(scales_w):
            if scales_w is not None:
                scales_w = torch_dialect.ConstantFloatOp(scales_w)
            else:
                scales_w = torch_dialect.ConstantNoneOp()
        else:
            scales_w = get_op_result_or_value(scales_w)
            assert str(scales_w.type) == '!torch.float', f'scales_w should a !torch.float but is {scales_w.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenUpsampleNearest2dOp, self).__init__(result_type, self_, output_size, scales_h, scales_w, loc=loc, ip=ip)
        
    
class Aten__Contains__IntListOp:
    def __init__(self, l: List[int], item: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = list(map(torch_dialect.ConstantIntOp, l))
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert str(l.type) == '!torch.list<int>', f'l should a !torch.list<int> but is {l.type}'
            
        if not is_mlir_value(item):
            item = torch_dialect.ConstantIntOp(item)
        else:
            item = get_op_result_or_value(item)
            assert str(item.type) == '!torch.int', f'item should a !torch.int but is {item.type}'
            
        super(Aten__Contains__IntListOp, self).__init__(l, item, loc=loc, ip=ip)
        
    
class AtenCatOp:
    def __init__(self, tensors: List["shark.Tensor"], dim: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tensors):
            tensors = torch_dialect.PrimListConstructOp(tensors)
        else:
            tensors = get_op_result_or_value(tensors)
            assert str(tensors.type) == '!torch.list<Tensor>', f'tensors should a !torch.list<Tensor> but is {tensors.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenCatOp, self).__init__(result_type, tensors, dim, loc=loc, ip=ip)
        
    
class AtenAppendTOp:
    def __init__(self, self_: List["shark.Tensor"], el: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.list<Tensor>', f'self_ should a !torch.list<Tensor> but is {self_.type}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'el should be a Tensor but is {el.type}'
        else:
            el = get_op_result_or_value(el)
            assert str(el.type).startswith("!torch.vtensor"), f'el should a Tensor but is {el.type}'
            
        result_type = Type.parse("!torch.list<Tensor>")
        super(AtenAppendTOp, self).__init__(result_type, self_, el, loc=loc, ip=ip)
        
    
class AtenAddTOp:
    def __init__(self, a: List["shark.Tensor"], b: List["shark.Tensor"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.list<Tensor>', f'a should a !torch.list<Tensor> but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.list<Tensor>', f'b should a !torch.list<Tensor> but is {b.type}'
            
        result_type = Type.parse("!torch.list<Tensor>")
        super(AtenAddTOp, self).__init__(result_type, a, b, loc=loc, ip=ip)
        
    
class AtenEqIntListOp:
    def __init__(self, a: List[int], b: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = list(map(torch_dialect.ConstantIntOp, a))
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.list<int>', f'a should a !torch.list<int> but is {a.type}'
            
        if not is_mlir_value(b):
            b = list(map(torch_dialect.ConstantIntOp, b))
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.list<int>', f'b should a !torch.list<int> but is {b.type}'
            
        super(AtenEqIntListOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenListTOp:
    def __init__(self, l: List["shark.Tensor"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert str(l.type) == '!torch.list<Tensor>', f'l should a !torch.list<Tensor> but is {l.type}'
            
        result_type = Type.parse("!torch.list<Tensor>")
        super(AtenListTOp, self).__init__(result_type, l, loc=loc, ip=ip)
        
    
class AtenSliceTOp:
    def __init__(self, l: List["shark.Tensor"], start: Optional[int], end: Optional[int], step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert str(l.type) == '!torch.list<Tensor>', f'l should a !torch.list<Tensor> but is {l.type}'
            
        if not is_mlir_value(start):
            if start is not None:
                start = torch_dialect.ConstantIntOp(start)
            else:
                start = torch_dialect.ConstantNoneOp()
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(end):
            if end is not None:
                end = torch_dialect.ConstantIntOp(end)
            else:
                end = torch_dialect.ConstantNoneOp()
        else:
            end = get_op_result_or_value(end)
            assert str(end.type) == '!torch.int', f'end should a !torch.int but is {end.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        result_type = Type.parse("!torch.list<Tensor>")
        super(AtenSliceTOp, self).__init__(result_type, l, start, end, step, loc=loc, ip=ip)
        
    
class AtenInsertTOp:
    def __init__(self, self_: List["shark.Tensor"], idx: int, el: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.list<Tensor>', f'self_ should a !torch.list<Tensor> but is {self_.type}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert str(idx.type) == '!torch.int', f'idx should a !torch.int but is {idx.type}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'el should be a Tensor but is {el.type}'
        else:
            el = get_op_result_or_value(el)
            assert str(el.type).startswith("!torch.vtensor"), f'el should a Tensor but is {el.type}'
            
        super(AtenInsertTOp, self).__init__(self_, idx, el, loc=loc, ip=ip)
        
    
class AtenNeIntListOp:
    def __init__(self, a: List[int], b: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = list(map(torch_dialect.ConstantIntOp, a))
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.list<int>', f'a should a !torch.list<int> but is {a.type}'
            
        if not is_mlir_value(b):
            b = list(map(torch_dialect.ConstantIntOp, b))
            b = torch_dialect.PrimListConstructOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.list<int>', f'b should a !torch.list<int> but is {b.type}'
            
        super(AtenNeIntListOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenAnyBoolOp:
    def __init__(self, self_: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = list(map(torch_dialect.ConstantBoolOp, self_))
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            # should be bool[]
            pass
            
        super(AtenAnyBoolOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenSortIntOp:
    def __init__(self, self_: List[int], reverse: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = list(map(torch_dialect.ConstantIntOp, self_))
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.list<int>', f'self_ should a !torch.list<int> but is {self_.type}'
            
        if not is_mlir_value(reverse):
            reverse = torch_dialect.ConstantBoolOp(reverse)
        else:
            reverse = get_op_result_or_value(reverse)
            assert str(reverse.type) == '!torch.bool', f'reverse should a !torch.bool but is {reverse.type}'
            
        super(AtenSortIntOp, self).__init__(self_, reverse, loc=loc, ip=ip)
        
    
class AtenAddStrOp:
    def __init__(self, a: str, b: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.str', f'a should a !torch.str but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantStrOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.str', f'b should a !torch.str but is {b.type}'
            
        super(AtenAddStrOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenEqStrOp:
    def __init__(self, a: str, b: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.str', f'a should a !torch.str but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantStrOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.str', f'b should a !torch.str but is {b.type}'
            
        super(AtenEqStrOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLenStrOp:
    def __init__(self, s: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(s):
            s = torch_dialect.ConstantStrOp(s)
        else:
            s = get_op_result_or_value(s)
            assert str(s.type) == '!torch.str', f's should a !torch.str but is {s.type}'
            
        super(AtenLenStrOp, self).__init__(s, loc=loc, ip=ip)
        
    
class AtenStrOp:
    def __init__(self, elem: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(elem):
            assert is_mlir_value(elem), f'elem should be a Tensor but is {elem.type}'
        else:
            elem = get_op_result_or_value(elem)
            assert str(elem.type).startswith("!torch.vtensor"), f'elem should a Tensor but is {elem.type}'
            
        super(AtenStrOp, self).__init__(elem, loc=loc, ip=ip)
        
    
class AtenJoinOp:
    def __init__(self, self_: str, values: List[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantStrOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.str', f'self_ should a !torch.str but is {self_.type}'
            
        if not is_mlir_value(values):
            values = list(map(torch_dialect.ConstantStrOp, values))
            values = torch_dialect.PrimListConstructOp(values)
        else:
            values = get_op_result_or_value(values)
            # should be str[]
            pass
            
        super(AtenJoinOp, self).__init__(self_, values, loc=loc, ip=ip)
        
    
class AtenFloatScalarOp:
    def __init__(self, a: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        super(AtenFloatScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenFloatStrOp:
    def __init__(self, a: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantStrOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.str', f'a should a !torch.str but is {a.type}'
            
        super(AtenFloatStrOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIntFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        super(AtenIntFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenIntScalarOp:
    def __init__(self, a: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        super(AtenIntScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class Aten__RangeLengthOp:
    def __init__(self, lo: int, hi: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(lo):
            lo = torch_dialect.ConstantIntOp(lo)
        else:
            lo = get_op_result_or_value(lo)
            assert str(lo.type) == '!torch.int', f'lo should a !torch.int but is {lo.type}'
            
        if not is_mlir_value(hi):
            hi = torch_dialect.ConstantIntOp(hi)
        else:
            hi = get_op_result_or_value(hi)
            assert str(hi.type) == '!torch.int', f'hi should a !torch.int but is {hi.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        super(Aten__RangeLengthOp, self).__init__(lo, hi, step, loc=loc, ip=ip)
        
    
class Aten__DeriveIndexOp:
    def __init__(self, index: int, start: int, step: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(index):
            index = torch_dialect.ConstantIntOp(index)
        else:
            index = get_op_result_or_value(index)
            assert str(index.type) == '!torch.int', f'index should a !torch.int but is {index.type}'
            
        if not is_mlir_value(start):
            start = torch_dialect.ConstantIntOp(start)
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(step):
            step = torch_dialect.ConstantIntOp(step)
        else:
            step = get_op_result_or_value(step)
            assert str(step.type) == '!torch.int', f'step should a !torch.int but is {step.type}'
            
        super(Aten__DeriveIndexOp, self).__init__(index, start, step, loc=loc, ip=ip)
        
    
class AtenGtIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenGtIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenGeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenLtIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenLeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenNeIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenEqIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenEqIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenFloordivIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenFloordivIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenRemainderIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenRemainderIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenRemainderScalarOp:
    def __init__(self, self_: "shark.Tensor", other: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(other):
            other = torch_dialect.ConstantNumberOp(other)
        else:
            other = get_op_result_or_value(other)
            assert str(other.type) in {'!torch.float', '!torch.int'}, f'other should a !torch.number but is {other.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenRemainderScalarOp, self).__init__(result_type, self_, other, loc=loc, ip=ip)
        
    
class AtenAddIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenAddIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenSubIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenMulIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenMulIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenDivIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenDivIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNegIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        super(AtenNegIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenLogIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        super(AtenLogIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenAddFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenAddFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenSubFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenMulFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenMulFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenDivFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenDivFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNegFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        super(AtenNegFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenEqFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenEqFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGtFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenGtFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenGeFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtFloatOp:
    def __init__(self, a: float, b: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantFloatOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.float', f'b should a !torch.float but is {b.type}'
            
        super(AtenLtFloatOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenLtFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenLtFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGeFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenGeFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenNeFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenGtFloatIntOp:
    def __init__(self, a: float, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(AtenGtFloatIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class Aten__And__BoolOp:
    def __init__(self, a: bool, b: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantBoolOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.bool', f'a should a !torch.bool but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantBoolOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.bool', f'b should a !torch.bool but is {b.type}'
            
        super(Aten__And__BoolOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenNeBoolOp:
    def __init__(self, a: bool, b: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantBoolOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.bool', f'a should a !torch.bool but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantBoolOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.bool', f'b should a !torch.bool but is {b.type}'
            
        super(AtenNeBoolOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class Aten__Is__Op:
    def __init__(self, self_: "shark.Tensor", obj: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            # should be t1
            pass
            
        if not is_mlir_value(obj):
            assert is_mlir_value(obj), f'obj should be a Tensor but is {obj.type}'
        else:
            obj = get_op_result_or_value(obj)
            # should be t2
            pass
            
        super(Aten__Is__Op, self).__init__(self_, obj, loc=loc, ip=ip)
        
    
class Aten__Isnot__Op:
    def __init__(self, self_: "shark.Tensor", obj: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            # should be t1
            pass
            
        if not is_mlir_value(obj):
            assert is_mlir_value(obj), f'obj should be a Tensor but is {obj.type}'
        else:
            obj = get_op_result_or_value(obj)
            # should be t2
            pass
            
        super(Aten__Isnot__Op, self).__init__(self_, obj, loc=loc, ip=ip)
        
    
class Aten__Not__Op:
    def __init__(self, self_: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = torch_dialect.ConstantBoolOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.bool', f'self_ should a !torch.bool but is {self_.type}'
            
        super(Aten__Not__Op, self).__init__(self_, loc=loc, ip=ip)
        
    
class AtenLenTOp:
    def __init__(self, a: List["shark.Tensor"], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.PrimListConstructOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.list<Tensor>', f'a should a !torch.list<Tensor> but is {a.type}'
            
        super(AtenLenTOp, self).__init__(a, loc=loc, ip=ip)
        
    
class Aten__Getitem__TOp:
    def __init__(self, list_: List["shark.Tensor"], idx: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(list_):
            list_ = torch_dialect.PrimListConstructOp(list_)
        else:
            list_ = get_op_result_or_value(list_)
            assert str(list_.type) == '!torch.list<Tensor>', f'list_ should a !torch.list<Tensor> but is {list_.type}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert str(idx.type) == '!torch.int', f'idx should a !torch.int but is {idx.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten__Getitem__TOp, self).__init__(result_type, list_, idx, loc=loc, ip=ip)
        
    
class Aten_SetItemTOp:
    def __init__(self, l: List["shark.Tensor"], idx: int, el: "shark.Tensor", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(l):
            l = torch_dialect.PrimListConstructOp(l)
        else:
            l = get_op_result_or_value(l)
            assert str(l.type) == '!torch.list<Tensor>', f'l should a !torch.list<Tensor> but is {l.type}'
            
        if not is_mlir_value(idx):
            idx = torch_dialect.ConstantIntOp(idx)
        else:
            idx = get_op_result_or_value(idx)
            assert str(idx.type) == '!torch.int', f'idx should a !torch.int but is {idx.type}'
            
        if not is_mlir_value(el):
            assert is_mlir_value(el), f'el should be a Tensor but is {el.type}'
        else:
            el = get_op_result_or_value(el)
            assert str(el.type).startswith("!torch.vtensor"), f'el should a Tensor but is {el.type}'
            
        result_type = Type.parse("!torch.list<Tensor>")
        super(Aten_SetItemTOp, self).__init__(result_type, l, idx, el, loc=loc, ip=ip)
        
    
class AtenDivOp:
    def __init__(self, a: "shark.Number", b: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) in {'!torch.float', '!torch.int'}, f'b should a !torch.number but is {b.type}'
            
        super(AtenDivOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenAddOp:
    def __init__(self, a: "shark.Number", b: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) in {'!torch.float', '!torch.int'}, f'b should a !torch.number but is {b.type}'
            
        super(AtenAddOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenSubOp:
    def __init__(self, a: "shark.Number", b: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantNumberOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) in {'!torch.float', '!torch.int'}, f'b should a !torch.number but is {b.type}'
            
        super(AtenSubOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class AtenCeilScalarOp:
    def __init__(self, a: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        super(AtenCeilScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenSqrtIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        super(AtenSqrtIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenBoolFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        super(AtenBoolFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenBoolIntOp:
    def __init__(self, a: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        super(AtenBoolIntOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenCeilFloatOp:
    def __init__(self, a: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantFloatOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.float', f'a should a !torch.float but is {a.type}'
            
        super(AtenCeilFloatOp, self).__init__(a, loc=loc, ip=ip)
        
    
class AtenNarrowOp:
    def __init__(self, self_: "shark.Tensor", dim: int, start: int, length: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(start):
            start = torch_dialect.ConstantIntOp(start)
        else:
            start = get_op_result_or_value(start)
            assert str(start.type) == '!torch.int', f'start should a !torch.int but is {start.type}'
            
        if not is_mlir_value(length):
            length = torch_dialect.ConstantIntOp(length)
        else:
            length = get_op_result_or_value(length)
            assert str(length.type) == '!torch.int', f'length should a !torch.int but is {length.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNarrowOp, self).__init__(result_type, self_, dim, start, length, loc=loc, ip=ip)
        
    
class AtenScalarImplicitOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(AtenScalarImplicitOp, self).__init__(a, loc=loc, ip=ip)
        
    
class Aten_SoftmaxBackwardDataOp:
    def __init__(self, grad_output: "shark.Tensor", output: "shark.Tensor", dim: int, input_dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'output should be a Tensor but is {output.type}'
        else:
            output = get_op_result_or_value(output)
            assert str(output.type).startswith("!torch.vtensor"), f'output should a Tensor but is {output.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(input_dtype):
            input_dtype = torch_dialect.ConstantIntOp(input_dtype)
        else:
            input_dtype = get_op_result_or_value(input_dtype)
            assert str(input_dtype.type) == '!torch.int', f'input_dtype should a !torch.int but is {input_dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_SoftmaxBackwardDataOp, self).__init__(result_type, grad_output, output, dim, input_dtype, loc=loc, ip=ip)
        
    
class AtenTanhBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", output: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'output should be a Tensor but is {output.type}'
        else:
            output = get_op_result_or_value(output)
            assert str(output.type).startswith("!torch.vtensor"), f'output should a Tensor but is {output.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenTanhBackwardOp, self).__init__(result_type, grad_output, output, loc=loc, ip=ip)
        
    
class AtenGeluBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", self_: "shark.Tensor", approximate: str, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(self_):
            assert is_mlir_value(self_), f'self_ should be a Tensor but is {self_.type}'
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type).startswith("!torch.vtensor"), f'self_ should a Tensor but is {self_.type}'
            
        if not is_mlir_value(approximate):
            approximate = torch_dialect.ConstantStrOp(approximate)
        else:
            approximate = get_op_result_or_value(approximate)
            assert str(approximate.type) == '!torch.str', f'approximate should a !torch.str but is {approximate.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenGeluBackwardOp, self).__init__(result_type, grad_output, self_, approximate, loc=loc, ip=ip)
        
    
class Aten_LogSoftmaxBackwardDataOp:
    def __init__(self, grad_output: "shark.Tensor", output: "shark.Tensor", dim: int, input_dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(output):
            assert is_mlir_value(output), f'output should be a Tensor but is {output.type}'
        else:
            output = get_op_result_or_value(output)
            assert str(output.type).startswith("!torch.vtensor"), f'output should a Tensor but is {output.type}'
            
        if not is_mlir_value(dim):
            dim = torch_dialect.ConstantIntOp(dim)
        else:
            dim = get_op_result_or_value(dim)
            assert str(dim.type) == '!torch.int', f'dim should a !torch.int but is {dim.type}'
            
        if not is_mlir_value(input_dtype):
            input_dtype = torch_dialect.ConstantIntOp(input_dtype)
        else:
            input_dtype = get_op_result_or_value(input_dtype)
            assert str(input_dtype.type) == '!torch.int', f'input_dtype should a !torch.int but is {input_dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(Aten_LogSoftmaxBackwardDataOp, self).__init__(result_type, grad_output, output, dim, input_dtype, loc=loc, ip=ip)
        
    
class AtenNativeLayerNormBackwardOp:
    def __init__(self, grad_out: "shark.Tensor", input: "shark.Tensor", normalized_shape: List[int], mean: "shark.Tensor", rstd: "shark.Tensor", weight: Optional["shark.Tensor"], bias: Optional["shark.Tensor"], output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_out):
            assert is_mlir_value(grad_out), f'grad_out should be a Tensor but is {grad_out.type}'
        else:
            grad_out = get_op_result_or_value(grad_out)
            assert str(grad_out.type).startswith("!torch.vtensor"), f'grad_out should a Tensor but is {grad_out.type}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(normalized_shape):
            normalized_shape = list(map(torch_dialect.ConstantIntOp, normalized_shape))
            normalized_shape = torch_dialect.PrimListConstructOp(normalized_shape)
        else:
            normalized_shape = get_op_result_or_value(normalized_shape)
            assert str(normalized_shape.type) == '!torch.list<int>', f'normalized_shape should a !torch.list<int> but is {normalized_shape.type}'
            
        if not is_mlir_value(mean):
            assert is_mlir_value(mean), f'mean should be a Tensor but is {mean.type}'
        else:
            mean = get_op_result_or_value(mean)
            assert str(mean.type).startswith("!torch.vtensor"), f'mean should a Tensor but is {mean.type}'
            
        if not is_mlir_value(rstd):
            assert is_mlir_value(rstd), f'rstd should be a Tensor but is {rstd.type}'
        else:
            rstd = get_op_result_or_value(rstd)
            assert str(rstd.type).startswith("!torch.vtensor"), f'rstd should a Tensor but is {rstd.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(bias):
            if bias is not None:
                assert is_mlir_value(bias), f'bias should be a Tensor but is {bias.type}'
            else:
                bias = torch_dialect.ConstantNoneOp()
        else:
            bias = get_op_result_or_value(bias)
            assert str(bias.type).startswith("!torch.vtensor"), f'bias should a Tensor but is {bias.type}'
            
        if not is_mlir_value(output_mask):
            output_mask = list(map(torch_dialect.ConstantBoolOp, output_mask))
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            # should be bool[]
            pass
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        super(AtenNativeLayerNormBackwardOp, self).__init__(result0_type, result1_type, result2_type, grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, loc=loc, ip=ip)
        
    
class AtenEmbeddingDenseBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", indices: "shark.Tensor", num_weights: int, padding_idx: int, scale_grad_by_freq: bool, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(indices):
            assert is_mlir_value(indices), f'indices should be a Tensor but is {indices.type}'
        else:
            indices = get_op_result_or_value(indices)
            assert str(indices.type).startswith("!torch.vtensor"), f'indices should a Tensor but is {indices.type}'
            
        if not is_mlir_value(num_weights):
            num_weights = torch_dialect.ConstantIntOp(num_weights)
        else:
            num_weights = get_op_result_or_value(num_weights)
            assert str(num_weights.type) == '!torch.int', f'num_weights should a !torch.int but is {num_weights.type}'
            
        if not is_mlir_value(padding_idx):
            padding_idx = torch_dialect.ConstantIntOp(padding_idx)
        else:
            padding_idx = get_op_result_or_value(padding_idx)
            assert str(padding_idx.type) == '!torch.int', f'padding_idx should a !torch.int but is {padding_idx.type}'
            
        if not is_mlir_value(scale_grad_by_freq):
            scale_grad_by_freq = torch_dialect.ConstantBoolOp(scale_grad_by_freq)
        else:
            scale_grad_by_freq = get_op_result_or_value(scale_grad_by_freq)
            assert str(scale_grad_by_freq.type) == '!torch.bool', f'scale_grad_by_freq should a !torch.bool but is {scale_grad_by_freq.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenEmbeddingDenseBackwardOp, self).__init__(result_type, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, loc=loc, ip=ip)
        
    
class AtenNativeBatchNormBackwardOp:
    def __init__(self, grad_out: "shark.Tensor", input: "shark.Tensor", weight: Optional["shark.Tensor"], running_mean: Optional["shark.Tensor"], running_var: Optional["shark.Tensor"], save_mean: Optional["shark.Tensor"], save_invstd: Optional["shark.Tensor"], train: bool, eps: float, output_mask: List[bool], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_out):
            assert is_mlir_value(grad_out), f'grad_out should be a Tensor but is {grad_out.type}'
        else:
            grad_out = get_op_result_or_value(grad_out)
            assert str(grad_out.type).startswith("!torch.vtensor"), f'grad_out should a Tensor but is {grad_out.type}'
            
        if not is_mlir_value(input):
            assert is_mlir_value(input), f'input should be a Tensor but is {input.type}'
        else:
            input = get_op_result_or_value(input)
            assert str(input.type).startswith("!torch.vtensor"), f'input should a Tensor but is {input.type}'
            
        if not is_mlir_value(weight):
            if weight is not None:
                assert is_mlir_value(weight), f'weight should be a Tensor but is {weight.type}'
            else:
                weight = torch_dialect.ConstantNoneOp()
        else:
            weight = get_op_result_or_value(weight)
            assert str(weight.type).startswith("!torch.vtensor"), f'weight should a Tensor but is {weight.type}'
            
        if not is_mlir_value(running_mean):
            if running_mean is not None:
                assert is_mlir_value(running_mean), f'running_mean should be a Tensor but is {running_mean.type}'
            else:
                running_mean = torch_dialect.ConstantNoneOp()
        else:
            running_mean = get_op_result_or_value(running_mean)
            assert str(running_mean.type).startswith("!torch.vtensor"), f'running_mean should a Tensor but is {running_mean.type}'
            
        if not is_mlir_value(running_var):
            if running_var is not None:
                assert is_mlir_value(running_var), f'running_var should be a Tensor but is {running_var.type}'
            else:
                running_var = torch_dialect.ConstantNoneOp()
        else:
            running_var = get_op_result_or_value(running_var)
            assert str(running_var.type).startswith("!torch.vtensor"), f'running_var should a Tensor but is {running_var.type}'
            
        if not is_mlir_value(save_mean):
            if save_mean is not None:
                assert is_mlir_value(save_mean), f'save_mean should be a Tensor but is {save_mean.type}'
            else:
                save_mean = torch_dialect.ConstantNoneOp()
        else:
            save_mean = get_op_result_or_value(save_mean)
            assert str(save_mean.type).startswith("!torch.vtensor"), f'save_mean should a Tensor but is {save_mean.type}'
            
        if not is_mlir_value(save_invstd):
            if save_invstd is not None:
                assert is_mlir_value(save_invstd), f'save_invstd should be a Tensor but is {save_invstd.type}'
            else:
                save_invstd = torch_dialect.ConstantNoneOp()
        else:
            save_invstd = get_op_result_or_value(save_invstd)
            assert str(save_invstd.type).startswith("!torch.vtensor"), f'save_invstd should a Tensor but is {save_invstd.type}'
            
        if not is_mlir_value(train):
            train = torch_dialect.ConstantBoolOp(train)
        else:
            train = get_op_result_or_value(train)
            assert str(train.type) == '!torch.bool', f'train should a !torch.bool but is {train.type}'
            
        if not is_mlir_value(eps):
            eps = torch_dialect.ConstantFloatOp(eps)
        else:
            eps = get_op_result_or_value(eps)
            assert str(eps.type) == '!torch.float', f'eps should a !torch.float but is {eps.type}'
            
        if not is_mlir_value(output_mask):
            output_mask = list(map(torch_dialect.ConstantBoolOp, output_mask))
            output_mask = torch_dialect.PrimListConstructOp(output_mask)
        else:
            output_mask = get_op_result_or_value(output_mask)
            # should be bool[]
            pass
            
        result0_type = Type.parse("!torch.vtensor")
        result1_type = Type.parse("!torch.vtensor")
        result2_type = Type.parse("!torch.vtensor")
        super(AtenNativeBatchNormBackwardOp, self).__init__(result0_type, result1_type, result2_type, grad_out, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps, output_mask, loc=loc, ip=ip)
        
    
class AtenNativeDropoutBackwardOp:
    def __init__(self, grad_output: "shark.Tensor", mask: "shark.Tensor", scale: float, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(grad_output):
            assert is_mlir_value(grad_output), f'grad_output should be a Tensor but is {grad_output.type}'
        else:
            grad_output = get_op_result_or_value(grad_output)
            assert str(grad_output.type).startswith("!torch.vtensor"), f'grad_output should a Tensor but is {grad_output.type}'
            
        if not is_mlir_value(mask):
            assert is_mlir_value(mask), f'mask should be a Tensor but is {mask.type}'
        else:
            mask = get_op_result_or_value(mask)
            assert str(mask.type).startswith("!torch.vtensor"), f'mask should a Tensor but is {mask.type}'
            
        if not is_mlir_value(scale):
            scale = torch_dialect.ConstantFloatOp(scale)
        else:
            scale = get_op_result_or_value(scale)
            assert str(scale.type) == '!torch.float', f'scale should a !torch.float but is {scale.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(AtenNativeDropoutBackwardOp, self).__init__(result_type, grad_output, mask, scale, loc=loc, ip=ip)
        
    
class PrimLayoutOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(PrimLayoutOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimTupleIndexOp:
    def __init__(self, tup: Any, i: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(tup):
            assert is_mlir_value(tup), f'tup should be a Tensor but is {tup.type}'
        else:
            tup = get_op_result_or_value(tup)
            assert str(tup.type) == '!torch.Any', f'tup should a !torch.Any but is {tup.type}'
            
        if not is_mlir_value(i):
            i = torch_dialect.ConstantIntOp(i)
        else:
            i = get_op_result_or_value(i)
            assert str(i.type) == '!torch.int', f'i should a !torch.int but is {i.type}'
            
        super(PrimTupleIndexOp, self).__init__(tup, i, loc=loc, ip=ip)
        
    
class PrimDtypeOp:
    def __init__(self, a: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        super(PrimDtypeOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimNumToTensorScalarOp:
    def __init__(self, a: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(PrimNumToTensorScalarOp, self).__init__(result_type, a, loc=loc, ip=ip)
        
    
class PrimMinSelfIntOp:
    def __init__(self, self_: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = list(map(torch_dialect.ConstantIntOp, self_))
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.list<int>', f'self_ should a !torch.list<int> but is {self_.type}'
            
        super(PrimMinSelfIntOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class PrimMinIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(PrimMinIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class PrimMaxSelfIntOp:
    def __init__(self, self_: List[int], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(self_):
            self_ = list(map(torch_dialect.ConstantIntOp, self_))
            self_ = torch_dialect.PrimListConstructOp(self_)
        else:
            self_ = get_op_result_or_value(self_)
            assert str(self_.type) == '!torch.list<int>', f'self_ should a !torch.list<int> but is {self_.type}'
            
        super(PrimMaxSelfIntOp, self).__init__(self_, loc=loc, ip=ip)
        
    
class PrimMaxIntOp:
    def __init__(self, a: int, b: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantIntOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) == '!torch.int', f'a should a !torch.int but is {a.type}'
            
        if not is_mlir_value(b):
            b = torch_dialect.ConstantIntOp(b)
        else:
            b = get_op_result_or_value(b)
            assert str(b.type) == '!torch.int', f'b should a !torch.int but is {b.type}'
            
        super(PrimMaxIntOp, self).__init__(a, b, loc=loc, ip=ip)
        
    
class PrimRaiseExceptionOp:
    def __init__(self, msg: str, cls: Optional[str], *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(msg):
            msg = torch_dialect.ConstantStrOp(msg)
        else:
            msg = get_op_result_or_value(msg)
            assert str(msg.type) == '!torch.str', f'msg should a !torch.str but is {msg.type}'
            
        if not is_mlir_value(cls):
            if cls is not None:
                cls = torch_dialect.ConstantStrOp(cls)
            else:
                cls = torch_dialect.ConstantNoneOp()
        else:
            cls = get_op_result_or_value(cls)
            assert str(cls.type) == '!torch.str', f'cls should a !torch.str but is {cls.type}'
            
        super(PrimRaiseExceptionOp, self).__init__(msg, cls, loc=loc, ip=ip)
        
    
class PrimUninitializedOp:
    def __init__(self, *, loc=None, ip=None):
        super(PrimUninitializedOp, self).__init__(loc=loc, ip=ip)
        
    
class PrimUncheckedCastOp:
    def __init__(self, x: "shark.Tensor", *, loc=None, ip=None):
        if not is_mlir_value(x):
            assert is_mlir_value(x), f'x should be a Tensor but is {x.type}'
        else:
            x = get_op_result_or_value(x)
            assert str(x.type).startswith("!torch.vtensor"), f'x should a Tensor but is {x.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(PrimUncheckedCastOp, self).__init__(result_type, x, loc=loc, ip=ip)
        
    
class PrimAbsScalarOp:
    def __init__(self, a: "shark.Number", *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            a = torch_dialect.ConstantNumberOp(a)
        else:
            a = get_op_result_or_value(a)
            assert str(a.type) in {'!torch.float', '!torch.int'}, f'a should a !torch.number but is {a.type}'
            
        super(PrimAbsScalarOp, self).__init__(a, loc=loc, ip=ip)
        
    
class PrimsConvertElementTypeOp:
    def __init__(self, a: "shark.Tensor", dtype: int, *, loc=None, ip=None):
        from torch_mlir.dialects import torch as torch_dialect
        
        if not is_mlir_value(a):
            assert is_mlir_value(a), f'a should be a Tensor but is {a.type}'
        else:
            a = get_op_result_or_value(a)
            assert str(a.type).startswith("!torch.vtensor"), f'a should a Tensor but is {a.type}'
            
        if not is_mlir_value(dtype):
            dtype = torch_dialect.ConstantIntOp(dtype)
        else:
            dtype = get_op_result_or_value(dtype)
            assert str(dtype.type) == '!torch.int', f'dtype should a !torch.int but is {dtype.type}'
            
        result_type = Type.parse("!torch.vtensor")
        super(PrimsConvertElementTypeOp, self).__init__(result_type, a, dtype, loc=loc, ip=ip)
        
    
