import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

# Function to generate random color
def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to create an image with random background and text
def create_random_frame(width, height):
    # Create a random background color
    background_color = random_color()
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = background_color

    # Add random text
    text = "Random Video"
    font = ImageFont.load_default()
    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    text_color = random_color()
    text_position = (random.randint(0, width // 2), random.randint(0, height // 2))

    draw.text(text_position, text, font=font, fill=text_color)
    return np.array(pil_image)

# Function to generate a random video
def generate_random_video(filename, width=640, height=480, frames=100, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video file
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for _ in range(frames):
        frame = create_random_frame(width, height)
        out.write(frame)

    out.release()

# Create a directory to save the videos
if not os.path.exists('random_videos'):
    os.makedirs('random_videos')

# Generate 10 random videos
for i in range(10):
    video_filename = f'random_videos/random_video_{i + 1}.avi'
    generate_random_video(video_filename)
    print(f'Generated {video_filename}')

print("All videos generated successfully!")
