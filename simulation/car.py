class Car:
    def __init__(self, start_point, end_point, time_left_at_this_road=0, time_total=0):
        self.start_point = start_point
        self.end_point = end_point
        self.at = start_point
        self.route = []
        self.time_left_at_this_road = time_left_at_this_road
        self.time_total = time_total
