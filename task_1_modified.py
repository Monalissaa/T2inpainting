def cows_eat_candies(cow_n, candy_m, cows_height, candy_heights):
    # all candies
    for i in range(candy_m):
        # last为0
        critical_point = 0
        # all cows
        for j in range(cow_n):
            if critical_point == candy_heights[i]:
                break
            if cows_height[j] <= critical_point:
                continue
            edible_candies = min(cows_height[j], candy_heights[i])
            eaten_candies = edible_candies - critical_point
            critical_point = min(cows_height[j], candy_heights[i])
            cows_height[j] += eaten_candies
    return cows_height

# input the value of cow_n, candy_m
cow_n, candy_m = map(int, input().split())
# input the list of initial height of cows_height
cows_height = list(map(int, input().split()))
# input the list of heights candy_heights
candy_heights = list(map(int, input().split()))

# process
cows_after_eat = cows_eat_candies(cow_n, candy_m, cows_height, candy_heights)
for cow in cows_after_eat:
    print(cow)
