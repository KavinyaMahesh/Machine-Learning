def vowel_consonants(strng):
    vowels_count=0
    cons_count=0
    vowels="aeiouAEIOU"
    for i in strng:
        if i in vowels:
           vowels_count+=1           
        else:
            cons_count+=1

            
    return vowels_count, cons_count


strng=input("Enter a string : ")
vowels,cons=vowel_consonants(strng)
print(f"Number of vowels: {vowels}")
print(f"Number of consonants: {cons}")




        





