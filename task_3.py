# Defining the structure 'node'
class DataPoint:
    def __init__(self, height, acceleration, time):
        self.height = height
        self.acceleration = acceleration
        self.time = time

# Function to sort the nodes
def sort_data_points(data_points):
    return sorted(data_points, key=lambda x: x.time, reverse=True)

# Main function
def main():
    num_tests = int(input().strip())
    for _ in range(num_tests):
        num_data_points = int(input().strip())
        data_points = [DataPoint(0, 0, 0) for _ in range(num_data_points)]
        height_values = list(map(int, input().strip().split()))
        acceleration_values = list(map(int, input().strip().split()))
        time_values = list(map(int, input().strip().split()))
        for i in range(num_data_points):
            data_points[i].height = height_values[i]
            data_points[i].acceleration = acceleration_values[i]
            data_points[i].time = time_values[i]
        
        data_points = sort_data_points(data_points)
        
        lower_bound, upper_bound = 0, 1e18
        is_valid = True
        for i in range(num_data_points-1):
            current_height = data_points[i].height - data_points[i+1].height
            current_acceleration = data_points[i+1].acceleration - data_points[i].acceleration
            if data_points[i].acceleration < data_points[i+1].acceleration:
                if current_height < 0:
                    continue
                current_time = current_height // current_acceleration + 1
                lower_bound = max(lower_bound, current_time)
            elif data_points[i].acceleration > data_points[i+1].acceleration:
                current_time = current_height // current_acceleration
                if current_height % current_acceleration == 0:
                    current_time -= 1
                upper_bound = min(upper_bound, current_time)
            else:
                if data_points[i].height < data_points[i+1].height:
                    continue
                else:
                    is_valid = False
        if lower_bound > upper_bound:
            print(-1)
        elif is_valid:
            print(lower_bound)
        else:
            print(-1)

# Calling the main function
if __name__ == "__main__":
    main()
