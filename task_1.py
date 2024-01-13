def input_function():
    # Number of cows and candies
    num_cows, num_candies = map(int, input().split())

    # List of cows and candies
    list_cows = list(map(int, input().split()))
    list_candies = list(map(int, input().split()))

    return num_cows, num_candies, list_cows, list_candies

# Call the function
num_cows, num_candies, list_cows, list_candies = input_function()

# For each candy
for candy_index in range(num_candies):
    # Initialize last_candy as 0
    last_candy = 0
    # Iterate over all cows
    for cow_index in range(num_cows):
        if last_candy == list_candies[candy_index]:
            break
        if last_candy >= list_cows[cow_index]:
            continue
        # At this point list_cows[cow_index] > last_candy, so we can directly eat the part [last_candy, min(list_cows[cow_index], list_candies[candy_index])]
        delta_candy = min(list_cows[cow_index], list_candies[candy_index]) - last_candy
        last_candy = min(list_cows[cow_index], list_candies[candy_index])
        list_cows[cow_index] += delta_candy

# Print the result
for cow in list_cows:
    print(cow)
