if __name__ == "__main__":
    pass

# Misc functions
def int_to_one_hot_vector(value, size, off_val=0, on_val=1):
# Size as 
    if int(value) < size:
        v = [off_val] * size
        v[int(value)] = on_val
        return v
    else:
        raise ValueError("Value is greater than size {} < {}".format(value, size))
    # (2,3) -> [0,0,1]    
def int_to_one_hot_vector_rev(value, size, off_value=0, on_val=1):
    v = int_to_one_hot_vector(value,size, off_value, on_val)
    v.reverse()
    return v


def int_to_binary(value,size=2): # int value to convert, size is length of array
    b_array = [int(x) for x in bin(value)[2:]] # Return as smallest binary number
    if(len(b_array) > size): 
        raise ValueError("Impossible to create a binary with size {} for binary({}) {}".format(size,value, b_array))
    if(len(b_array) < size):
        trailing_zeros = size - len(b_array) # check if binary is smaller than we require in digits.
        for zero in range(trailing_zeros):
            b_array.insert(0,0) # Add trailing zeros 01 -> x*0+01
    return b_array   

# Use to represent states and PID, (2,2) -> [0,1], (1,2) -> [1,0]
def int_to_binary_rev(value,size=2):
    v = int_to_binary(value, size)
    v.reverse()
    return v

def normalize_array(data, x_min=1):
    """ 
        min_max scaling: x_i = x_i-min(x)/(max(x) - min(x)) 
        z-score : x = x_i - sd/mean(data) 
    """
    #x_max = max(data)    
    #x_min = min(data)
    #data = [(x-x_min)/(x_max-x_min) for x in data]
    mean = sum(data)
    return [x/mean for x in data]

def min_max_scaling(data):
    x_max = max(data)    
    x_min = min(data)
    return [(x-x_min)/(x_max-x_min) for x in data]
