def one_hot_encoding(categories):
    unique_categories = list(set(categories))
    num_categories = len(unique_categories)
    encoded_matrix = []

    for category in categories:
        encoded_row = [0] * num_categories
        category_index = unique_categories.index(category)
        encoded_row[category_index] = 1
        encoded_matrix.append(encoded_row)

    return encoded_matrix

categories = ['cat', 'dog', 'cat', 'bird', 'dog', 'dog', 'bird']
encoded_matrix = one_hot_encoding(categories)
print("Original Categories:", categories)
print("Encoded Matrix:")
for row in encoded_matrix:
    print(row)
