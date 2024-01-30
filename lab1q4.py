def transpose_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    transposed_matrix = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def input_matrix_from_user(rows, cols):
    matrix = []
    print(f"Enter the elements of the matrix ({rows}x{cols}):")
    for i in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            print(f"Error: Each row should contain {cols} elements.")
            return None
        matrix.append(row)
    return matrix

rows = int(input("Enter the number of rows for the matrix: "))
cols = int(input("Enter the number of columns for the matrix: "))
matrix = input_matrix_from_user(rows, cols)

if matrix is not None:
  
    transposed_matrix = transpose_matrix(matrix)

    print("Original Matrix:")
    for row in matrix:
        print(row)

    print("\nTransposed Matrix:")
    for row in transposed_matrix:
        print(row)
