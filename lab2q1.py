import math

def euclidean_distance(vector1, vector2):
   
    sum_of_squares = 0
    for i in range(len(vector1)):
        sum_of_squares += (vector1[i] - vector2[i])**2

    return math.sqrt(sum_of_squares)

def manhattan_distance(vector1, vector2):
    
    sum_of_distances = 0
    for i in range(len(vector1)):
        sum_of_distances += abs(vector1[i] - vector2[i])

    return sum_of_distances

vector_a = [1, 2, 3]
vector_b = [4, 5, 6]

euclidean_result = euclidean_distance(vector_a, vector_b)
manhattan_result = manhattan_distance(vector_a, vector_b)

print(f"Euclidean Distance: {euclidean_result}")
print(f"Manhattan Distance: {manhattan_result}")


