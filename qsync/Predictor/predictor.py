def linear_func(x, a, b):
    return a*x + b

class AnalyticModel:
    def __init__(self, func=linear_func, params=None, half_to_int_params=None, float_to_int_params=None \
    ,int_to_half_params_scale=None,  int_to_float_params_scale=None, float_scale=None, \
    half_to_int_params_channel=None, float_to_int_params_channel=None, int_to_half_params_channel=None, int_to_float_params_channel=None, \
    float_channel=None, comm_all_to_tree=None,
    ):
        assert params is not None, "parmas must be set"
        self.func = func
        self.params = params
        self.half_to_int_params = half_to_int_params
        self.float_to_int_params = float_to_int_params

        self.int_to_half_params_scale = int_to_half_params_scale
        self.int_to_float_params_scale = int_to_float_params_scale
        self.float_scale = float_scale

        self.half_to_int_params_channel = half_to_int_params_channel
        self.float_to_int_params_channel = float_to_int_params_channel
        self.int_to_half_params_channel = int_to_half_params_channel
        self.int_to_float_params_channel = int_to_float_params_channel
        self.float_channel = float_channel


        self.comm_all_to_tree = comm_all_to_tree
    
    def predict(self, input):
        return self.func(input, *self.params)
    
    def predict_half_to_int(self, input):
        return self.func(input, *self.half_to_int_params)
    
    def predict_float_to_int(self, input):
        return self.func(input, *self.float_to_int_params)
    
    def predictint_to_half_scale(self, input):
        return self.func(input, *self.int_to_half_params_scale)
    
    def predictint_to_float_scale(self, input):
        return self.func(input, *self.int_to_float_params_scale)
    
    def predict_float_scale(self, input):
        return self.func(input, *self.float_scale)
    
    def predict_with_bit(self, bit_a, bit_b, nums):
        if bit_a == bit_b:
            return 0
        if (bit_a == 16 and bit_b == 32) or (bit_a == 32 and bit_b == 16):
            return self.predict(nums)
        elif bit_a == 16 and bit_b == 8:
            return self.predict_half_to_int(nums)
        elif bit_a == 32 and bit_b == 8:
            return self.predict_float_to_int(nums)
        elif bit_a == 8 and bit_b == 32:
            return self.predictint_to_float_scale(nums)
        elif bit_a == 8 and bit_b == 16:
            return self.predictint_to_half_scale(nums)
    
    # channel-wise version
    def predict_half_to_int_channel(self, input):
        return self.func(input, *self.half_to_int_params_channel)
    
    def predict_float_to_int_channel(self, input):
        return self.func(input, *self.float_to_int_params_channel)
    
    def predict_int_to_half_channel(self, input):
        return self.func(input, *self.int_to_half_params_channel)
    
    def predict_int_to_float_channel(self, input):
        return self.func(input, *self.int_to_float_params_channel)
    
    def predict_float_channel(self, input):
        return self.func(input, *self.float_channel)
    
    def predict_with_bit_channel(self, bit_a, bit_b, nums):
        if bit_a == bit_b:
            return 0
        if (bit_a == 16 and bit_b == 32) or (bit_a == 32 and bit_b == 16):
            return self.predict(nums)
        elif bit_a == 16 and bit_b == 8:
            return self.predict_half_to_int_channel(nums)
        elif bit_a == 32 and bit_b == 8:
            return self.predict_float_to_int_channel(nums)
        elif bit_a == 8 and bit_b == 32:
            return self.predict_int_to_float_channel(nums)
        elif bit_a == 8 and bit_b == 16:
            return self.predict_int_to_half_channel(nums)
    
    def predict_comm_add_cost(self, input):
        return self.func(input, *self.comm_all_to_tree)
    
    

# from analyse result of casting_cost.py
ana_predictor = AnalyticModel(params=[2.8777962860836457e-08, 0.0010094620016573655], \
 half_to_int_params=[1.285224864545341e-07, 0.0004578774832783733],\
 float_to_int_params=[1.4182117284950922e-07, 0.012216083171593976],\
 int_to_half_params_scale=[2.6165703850142927e-08, 0.00421425732078102],  \
 int_to_float_params_scale=[2.7271576955792816e-08, -0.002546326253177766], \
 float_scale=[2.6561163960098748e-08, 0.0061388293144670515], \
 half_to_int_params_channel=[1.1225706580917037e-07, 0.02539318282923971], \
 float_to_int_params_channel=[1.7104348229978765e-07, 0.0290379427540817], \
 int_to_half_params_channel=[5.935936745976087e-08, 0.008540220728762678], \
 int_to_float_params_channel=[3.3658425464562905e-08, 0.0048838988338678335], \
 float_channel=[1.7126423078718732e-08, 0.006201549706716251],\
 comm_all_to_tree=[1.6799620753813687e-06, -4.319022821824095] 
 )