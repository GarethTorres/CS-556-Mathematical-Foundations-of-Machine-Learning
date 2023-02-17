#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np


# In[71]:


firstarray = np.array([[0,1,3,7],
                      [4,8,0,2]])


# In[72]:


firstarray_shape = firstarray.shape


# In[73]:


print("The array is:\n", firstarray)


# In[74]:


print("The shape of the array is:", firstarray_shape)
#question 1#


# In[75]:


non_zero_elements = firstarray.ravel()[np.flatnonzero(firstarray)]


# In[76]:


print("The number of non zero elements in the array are:", non_zero_elements)
#question 2.1#


# In[77]:


array_after_delete = np.delete(firstarray, [2,4],axis=None)


# In[78]:


array_after_delete.resize((2, 3))


# In[79]:


print("The array after deleting the elements at index 2 and 4 is:\n", array_after_delete)
#question 2.2#


# In[80]:


array_after_insert_step1 = np.insert(array_after_delete,2,5,axis=None)


# In[81]:


array_after_insert = np.insert(array_after_insert_step1,4,9,axis=None)


# In[82]:


array_after_insert.resize((2, 4))


# In[83]:


print("The array after inserting 5 and 9 is:\n", array_after_insert)
#question 2.3#


# In[84]:


mag_of_vector1 = np.sqrt(sum(i**2 for i in [0,1,3,7]))


# In[85]:


print("The magnitude of vector 1 is =", mag_of_vector1)


# In[86]:


mag_of_vector2 = np.sqrt(sum(i**2 for i in [4,8,0,2]))


# In[87]:


print("The magnitude of vector 2 is =", mag_of_vector2)
#question 2.4#


# In[88]:


matrix1 = np.matrix([[4,3],[2,6]])


# In[89]:


matrix2 = np.matrix([[6,5],[7,8]])


# In[90]:


print("The first matrix is:\n", matrix1)


# In[91]:


print("\nThe second matrix is:\n", matrix2)
#question 3.1#


# In[92]:


det_1 = int(np.linalg.det(matrix1))


# In[93]:


det_2 = int(np.linalg.det(matrix2))


# In[94]:


print("The determinant of matrix 1 is:", det_1)


# In[95]:


print("The determinant of matrix 2 is:", det_2)
#question 3.2#


# In[96]:


dot_product = np.dot(matrix1,matrix2)


# In[97]:


print("The dot product of the 2 matrices is:\n", dot_product)
#question 3.3#


# In[98]:


sorted_dot_product = np.sort(dot_product)


# In[99]:


print("The sorted dot product is:\n", sorted_dot_product)
#question 3.4#


# In[100]:


transpose_1 = np.transpose(matrix1)


# In[101]:


transpose_2 = np.transpose(matrix2)


# In[102]:


print("The transpose of the first matrix is:\n", transpose_1)


# In[103]:


print("\nThe transpose of the second matrix is:\n", transpose_2)
#question 3.5#


# In[104]:


new_array = np.arange(9)


# In[105]:


print("The new 1-D array is:\n", new_array)
#question 4.1#


# In[106]:


cube_array = np.power(new_array,3)


# In[107]:


print("The cube of each number of the original array is:\n", cube_array)
#question 4.2#


# In[108]:


thirdarray = np.array([[20, 10, 40], 
                       [30, 70, 50]])


# In[109]:


print("The array is:\n", thirdarray)
#question 4.3#


# In[110]:


mean_rows = thirdarray.mean(axis=1)


# In[111]:


mean_columns = thirdarray.mean(axis=0)


# In[112]:


print("The means of the rows are:\n", mean_rows)


# In[113]:


print("\nThe means of the columns are:\n", mean_columns)
#question 4.4#


# In[114]:


mat1 = np.matrix([[1,2,0],[3,-1,2],[-2,3,-2]])


# In[115]:


mat2 = np.matrix([[ 5, -6, 6],[ 0, 7, 3],[-1, 8, 1]])


# In[116]:


print("The first matrix is:\n", mat1)


# In[117]:


print("\nThe second matrix is:\n", mat2)
#question 5.1#


# In[118]:


print('Is the mat1 are invertible?: ', (int(np.linalg.det(mat1))!=0))


# In[119]:


print('Is the mat2 are invertible?: ', (int(np.linalg.det(mat2))!=0))
#question 5.2#


# In[120]:


from numpy.linalg import inv


# In[121]:


if(int(np.linalg.det(mat1))!=0):
    inv1 = np.linalg.inv(mat1)
else:
    print("can't be inverse")


# In[122]:


inv2 = np.linalg.inv(mat2)


# In[123]:


print("Inverse for mat2 is:", inv2)
#question 5.3#


# In[124]:


first_col = inv2[...,0]


# In[125]:


print("The first column of the inverted matrix is:", first_col)
#question 5.4#


# In[126]:


m1 = np.matrix([[5, -10],[-4, 9]],dtype=np.float)


# In[127]:


m2 = np.matrix([[-5, -10],[11, 14]],dtype=np.float)


# In[128]:


print("The first matrix is:\n", m1)


# In[129]:


print("\nThe second matrix is:\n", m2)
#question 6.1#


# In[130]:


matrix_mul = np.dot(m1,m2)


# In[131]:


print("m1 x m2 = \n", matrix_mul)
#question 6.2#


# In[132]:


elem_mul = np.multiply(m1,m2)


# In[133]:


print("Element wise product = ", elem_mul)
#question 6.3#


# In[134]:


sqr_root = np.sqrt([[43,9],[22,34]])


# In[135]:


print("The square root is = ", sqr_root)
#question 6.4#


# In[31]:


def reshape(L,m,n):
# reshape L into a 2d list with m-by-n elements
    return [L[i*n:i*n+n] for i in range(0,m)]
#part2.1#
example1 = [0,1,2,3,4,5,6,7,8,9,10,11,12]
example2 = 4
example3 = 3
print(reshape(example1,example2,example3))


# In[5]:


def innprod(u, v):
    # inner product of u and v
    length_of_u = len(u)
    length_of_v = len(v)
    if(length_of_u!=length_of_u or length_of_u==0):
        raise Exception("no inner product of u and v")
    res = 0
    for i in range(length_of_u):
        res+=(u[i] * v[i])
    return res
#part2.2#
example11 =[89,2,3123,456]
example22 =[9,62,72,822]
print(innprod(example11,example22))


# In[7]:


def transpose(M):
    # return transpose matrix
    Number_of_row = len(M)
    if(Number_of_row==0):
        raise Exception("can't be transpose")
    Number_of_column = len(M[0])
    res=[]
    for r in range(Number_of_column):
        res.append([])
    for r in range(Number_of_row):
        for c in range(Number_of_column):
            res[c].append(M[r][c])
    return res
#part2.3#
example111 =[
    [95,542,355,4],
    [254,1545,2,1545]
]
print(transpose(example111))


# In[4]:


def mult(A, B):
    # product of matrices
    length_of_A = len(A)
    length_of_B = len(B)
    if(length_of_A!=length_of_A or length_of_A==0):
        raise Exception("no product of matrices")
    res = []
    for i in range(length_of_A):
        res.append(A[i] * B[i])
    return res
#part2.4#
example1111 =[545,54,3,4536]
example2222 =[6,562,65,89]
print(mult(example1111,example2222))

