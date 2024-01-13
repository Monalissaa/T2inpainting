# Input the number of cow and fever status
cow_number = int(input())
flu_status = input()

# Initialize infection_interval and counter
infection_interval = []
counter = 0

# Use split function to get the length of each '1' sequence
infection_interval = [len(i) for i in flu_status.split('0') if i]

# If the length of infection_interval is less than or equal to 1, output the length of infection_interval
if len(infection_interval) <= 1:
    print(len(infection_interval))
else:
    # Initialize minimum_flu_days
    minimum_flu_days = 100000000

    # Calculate how many days each all-1 interval can correspond to the most illness
    for idx in range(len(infection_interval)):
        # Special judgment at the beginning and end
        if idx == 0 and flu_status[0] == '1':
            now_days = infection_interval[idx] - 1
        elif idx == len(infection_interval) - 1 and flu_status[-1] == '1':
            now_days = infection_interval[idx] - 1
        else:
            now_days = (infection_interval[idx] - 1) // 2
        minimum_flu_days = min(minimum_flu_days, now_days)

    # Initialize ideal_flu_cows
    ideal_flu_cows = 0

    # Now that minimum_flu_days have passed, calculate the initial number of sick cows
    for idx in range(len(infection_interval)):
        # The length of 1 for infection_interval[idx], after minimum_flu_days, constitutes the minimum number of sick cows
        sick_cows = infection_interval[idx] // (2 * minimum_flu_days + 1)
        ideal_flu_cows += sick_cows
        # Round up
        if infection_interval[idx] % (2 * minimum_flu_days + 1):
            additional_sick_cows = 1
            ideal_flu_cows += additional_sick_cows


    # Output the ideal_flu_cows
    print(ideal_flu_cows)
