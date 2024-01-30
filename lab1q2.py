def input_matrix(rows, cols):
    matrix = []
    print(f"Enter the elements of the matrix ({rows}x{cols}):")
    for i in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            print(f"Error: Each row should contain {cols} elements.")
            return None
        matrix.append(row)
    return matrix

def matrix_multiply(A, B):
   
    if len(A[0]) != len(B):
        return "Matrices are not multipliable."

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


rows_A = int(input("Enter the number of rows for matrix A: "))
cols_A = int(input("Enter the number of columns for matrix A: "))
matrix_A = input_matrix(rows_A, cols_A)

rows_B = int(input("Enter the number of rows for matrix B: "))
cols_B = int(input("Enter the number of columns for matrix B: "))
matrix_B = input_matrix(rows_B, cols_B)

result_matrix = matrix_multiply(matrix_A, matrix_B)

if isinstance(result_matrix, str):
    print(result_matrix)
else:
    print("Product of A and B:")
    for row in result_matrix:
        print(row)
