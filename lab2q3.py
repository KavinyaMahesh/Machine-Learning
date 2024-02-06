def label_encoding(categories):
    unique_categories = list(set(categories))
    label_mapping = {category: index for index, category in enumerate(unique_categories)}
    encoded_labels = [label_mapping[category] for category in categories]
    return encoded_labels

categories = ['cat', 'dog', 'cat', 'bird', 'dog', 'dog', 'bird']
encoded_labels = label_encoding(categories)
print("Original Categories:", categories)
print("Encoded Labels:", encoded_labels)
