def common_elements_count(list1, list2):
    common_elements = set(list1) & set(list2)
    return len(common_elements)


list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]

common_count = common_elements_count(list1, list2)
print(f"Number of common elements: {common_count}")

print("Hello")
