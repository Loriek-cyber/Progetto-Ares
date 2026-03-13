import struct


def read_fast_lane(file_path):
    with open(file_path, "rb") as f:
        # The first few bytes usually contain the number of points (header)
        header = f.read(4)
        num_points = struct.unpack('i', header)[0]

        points = []
        for _ in range(num_points):
            # Each point typically contains:
            # Position (x, y, z), Velocity, and Track Width (Left/Right)
            # This follows a specific byte-pattern (e.g., 'fffffff')
            data = f.read(28)  # 7 floats * 4 bytes
            point = struct.unpack('fffffff', data)
            points.append(point)

    return points


# Usage
path = "C:/Users/Arjel/Giochi/Assetto Corsa/content/tracks/monza/ai/fast_lane.ai"
trajectory = read_fast_lane(path)
print(f"Loaded {len(trajectory)} points from the ideal line!")