# Misc functions

def int_to_one_hot_vector(value, size, off_val=0, on_val=1):
    # Size as 
    if int(value) < size:
        v = [off_val] * size
        v[int(value)] = on_val
        return v
def int_to_binary(value,size=3): # int value to convert, size is length of array
    if int(value) <= size: # 0-value binary
        b_array = [int(x) for x in bin(value)[2:]] # Return as smallest binary number
        if(len(b_array) < size):
            trailing_zeros = size - len(b_array) 
            for zero in range(trailing_zeros):
                b_array.append(0)
        return b_array   


print(int_to_one_hot_vector(0,2))
print(int_to_one_hot_vector(1,2))
print(int_to_one_hot_vector(2,2))

print(int_to_binary(0))
print(int_to_binary(1))
print(int_to_binary(2))
