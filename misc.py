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
def int_to_one_hot_vector_rev(value, size, off_value=0, on_val=1):
    v = int_to_one_hot_vector(value,size, off_value, on_val)
    v.reverse()
    return v


def int_to_binary(value,size=2): # int value to convert, size is length of array
    b_array = [int(x) for x in bin(value)[2:]] # Return as smallest binary number
    if(len(b_array) > size): 
        raise ValueError("Impossible to create a binary with size {} for binary({}) {}".format(size,value, b_array))
    if(len(b_array) < size):
        trailing_zeros = size - len(b_array) 
        for zero in range(trailing_zeros):
            b_array.insert(0,0)

    return b_array   

# Use to represent states and PID
def int_to_binary_rev(value,size=2):
    v = int_to_binary(value, size)
    v.reverse()
    return v
