# Number of tests
num_t = int(input().strip())
for _ in range(num_t):
    # Number of data points
    num_dp = int(input().strip())
    # Initialize data points
    dp = [[0, 0, 0] for _ in range(num_dp)]
    # Input height, acceleration, and time values
    h_vals = list(map(int, input().strip().split()))
    a_vals = list(map(int, input().strip().split()))
    t_vals = list(map(int, input().strip().split()))
    for i in range(num_dp):
        dp[i][0] = h_vals[i]
        dp[i][1] = a_vals[i]
        dp[i][2] = t_vals[i]
    
    # Sort data points based on time
    dp.sort(key=lambda x: x[2], reverse=True)
    
    # Initialize lower and upper bounds
    lb, ub = 0, 1e18
    # Flag to check validity
    v_flag = True
    for i in range(num_dp-1):
        # Calculate current height and acceleration
        c_h = dp[i][0] - dp[i+1][0]
        c_a = dp[i+1][1] - dp[i][1]
        if dp[i+1][1] > dp[i][1]:
            if c_h < 0:
                continue
            c_t = c_h // c_a + 1
            lb = max(lb, c_t)
        elif dp[i+1][1] < dp[i][1]:
            c_t = c_h // c_a
            if c_h % c_a == 0:
                c_t -= 1
            ub = min(ub, c_t)
        else:
            if dp[i][0] < dp[i+1][0]:
                continue
            else:
                v_flag = False
    print(lb if lb <= ub else -1)