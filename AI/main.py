import os
import random
import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import cvzone
import pygame
from PIL import Image, ImageDraw, ImageFont

# Load the pixelated font
pixel_font_path = 'Fonts/pixel_font.ttf'  # Path to your .ttf file
try:
    pixel_font = ImageFont.truetype(pixel_font_path, 100)  # Set font size
except IOError:
    print("Error: Pixel font file not found!")
    exit()

# Function for overlaying text with pixel font
def overlay_top_centered_text_with_pixel_font(image, text, font, color, vertical_offset=50, border_color=(0, 0, 0), border_thickness=2):
    """
    Displays top-centered text using a pixel font with a border onto an OpenCV image.
    """
    # Convert OpenCV image to PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Calculate text size using getbbox
    bbox = font.getbbox(text)  # (left, top, right, bottom)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Get image dimensions
    image_width, _ = pil_image.size

    # Calculate position for top-centered alignment
    position = (
        (image_width - text_width) // 2,  # Center horizontally
        vertical_offset  # Fixed vertical position
    )

    # Draw the border by rendering the text multiple times around its position
    for dx in range(-border_thickness, border_thickness + 1):
        for dy in range(-border_thickness, border_thickness + 1):
            if dx != 0 or dy != 0:  # Avoid drawing the main text position
                draw.text((position[0] + dx, position[1] + dy), text, font=font, fill=border_color)

    # Draw the main text
    draw.text(position, text, font=font, fill=color)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Initialize pygame mixer for sounds
pygame.init()
pygame.mixer.init()

# Load sound effects
bg_music = 'SFX/bg_music.mp3'
fruit_sound = 'SFX/fruit_eaten.wav'
bomb_sound = 'SFX/bomb_eaten.wav'
game_over_sound = 'SFX/game_over.wav'
speed_increase_sound = 'SFX/speed_increase.wav'

# Check if sound files exist
if not all(os.path.exists(sound) for sound in [bg_music, fruit_sound, bomb_sound, game_over_sound]):
    print("Error: One or more sound files are missing!")
    exit()

# Load individual sound effects
fruit_eaten_sfx = pygame.mixer.Sound(fruit_sound)
bomb_eaten_sfx = pygame.mixer.Sound(bomb_sound)
game_over_sfx = pygame.mixer.Sound(game_over_sound)
speed_increase_sfx = pygame.mixer.Sound(speed_increase_sound)
# Set sound effect volumes
fruit_eaten_sfx.set_volume(0.7)
bomb_eaten_sfx.set_volume(0.7)
game_over_sfx.set_volume(0.8)
speed_increase_sfx.set_volume(0.7)

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 1080)  # Set height

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize FaceMesh detector
detector = FaceMeshDetector(maxFaces=2)
idList = [0, 17, 78, 292]  # Key points for detecting mouth

# Import images for eatable and non-eatable objects
folderEatable = 'Objects/eatable'
folderNonEatable = 'Objects/noneatable'

if not os.path.exists(folderEatable) or not os.path.exists(folderNonEatable):
    print("Error: Required object folders are missing!")
    exit()

# Load eatable and non-eatable objects
eatables = [
    cv2.imread(f'{folderEatable}/{obj}', cv2.IMREAD_UNCHANGED)
    for obj in os.listdir(folderEatable) if obj.endswith(('png', 'jpg', 'jpeg'))
]
nonEatables = [
    cv2.imread(f'{folderNonEatable}/{obj}', cv2.IMREAD_UNCHANGED)
    for obj in os.listdir(folderNonEatable) if obj.endswith(('png', 'jpg', 'jpeg'))
]

# Load health images
health_folder = 'Health'
maxHealth = 3
health_images = [
    cv2.imread(f'{health_folder}/health_{i}.png', cv2.IMREAD_UNCHANGED)
    for i in range(maxHealth + 1)
]

if any(img is None for img in health_images):
    print("Error: Health images not found!")
    exit()

# Class for falling objects
class FallingObject:
    """
    Represents a falling object in the game.
    """
    def __init__(self, img, pos, speed, isEatable):
        self.img = cv2.resize(img, (80, 80))
        self.pos = pos
        self.speed = speed
        self.isEatable = isEatable

# Function to reset object position and type
def resetObject(speed):
    """
    Resets the position and type of a falling object.
    """
    pos = [random.randint(100, 1180), 0]
    isEatable = random.randint(0, 2) != 0
    currentObject = random.choice(eatables if isEatable else nonEatables)
    return FallingObject(currentObject, pos, speed, isEatable)

# Home screen display function
def homeScreen():
    """
    Displays the home screen image and waits for the user to start the game.
    """
    home_image_path = 'Screens/home_screen.png'
    home_img = cv2.imread(home_image_path)

    if home_img is None:
        print("Error: Home screen image not found!")
        exit()

    cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow("Image", home_img)
        key = cv2.waitKey(1)
        if key == ord('s'):  # Start the game
            break
        elif key == 27:  # ESC key to exit
            exit()

# Game Over screen display function
def gameOverScreen(score):
    """
    Displays the game over screen with the player's score and options to restart or exit.
    """
    game_over_image_path = 'Screens/game_over_screen.png'
    game_over_img = cv2.imread(game_over_image_path)

    if game_over_img is None:
        print("Error: Game Over screen image not found!")
        exit()

    # Add score dynamically to the image
    pil_image = Image.fromarray(cv2.cvtColor(game_over_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    score_text = f"{score}"

    # Load and set the font for dynamic text
    font_path = 'Fonts/pixel_font.ttf'
    try:
        font = ImageFont.truetype(font_path, 150)
    except IOError:
        print("Error: Font file not found!")
        exit()

    # Calculate the position to center the score
    bbox = font.getbbox(score_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    img_width, img_height = pil_image.size

    horizontal_offset = 100
    vertical_offset = 590

    x_position = ((img_width - text_width) // 2) + horizontal_offset
    y_position = img_height - vertical_offset

    # Draw the black border
    border_offset = 5
    for offset_x in range(-border_offset, border_offset + 1):
        for offset_y in range(-border_offset, border_offset + 1):
            if offset_x != 0 or offset_y != 0:
                draw.text((x_position + offset_x, y_position + offset_y), score_text, font=font, fill=(0, 0, 0))

    # Draw the main score text in white
    draw.text((x_position, y_position), score_text, font=font, fill=(255, 255, 255))

    game_over_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cv2.imshow("Image", game_over_img)
        key = cv2.waitKey(1)
        if key == ord('r'):  # Restart the game
            return 'restart'
        elif key == 27:  # ESC key to exit
            return 'exit'

# Initialize game parameters
numObjects, maxObjects, initialSpeed = 1, 8, 7
speed, count, health = initialSpeed, 0, maxHealth
objects = [resetObject(speed) for _ in range(numObjects)]
gameOver = False

# Call home screen
homeScreen()

# Play background music in a loop
pygame.mixer.music.load(bg_music)
pygame.mixer.music.set_volume(1)
pygame.mixer.music.play(-1)

# Load speed images
speed_folder = 'Speed'
speed_images = {}

for speed_level in range(7, 100, 2):
    img_path = f"{speed_folder}/speed_{speed_level}.png"
    if os.path.exists(img_path):
        speed_images[speed_level] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if not speed_images:
    print("Error: Speed images not found!")
    exit()

# Load emoji images
emoji_folder = 'Emojis'
open_mouth_emoji = cv2.imread(f'{emoji_folder}/open.png', cv2.IMREAD_UNCHANGED)
closed_mouth_emoji = cv2.imread(f'{emoji_folder}/closed.png', cv2.IMREAD_UNCHANGED)

if open_mouth_emoji is None or closed_mouth_emoji is None:
    print("Error: Emoji images not found!")
    exit()

# Resize emoji images to the same size as falling objects
emoji_size = (80, 80)
open_mouth_emoji = cv2.resize(open_mouth_emoji, emoji_size)
closed_mouth_emoji = cv2.resize(closed_mouth_emoji, emoji_size)

played_game_over_sfx = False

# Main game loop
while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    img = cv2.flip(img, 1)

    if not gameOver:
        # Detect face and overlay objects
        img, faces = detector.findFaceMesh(img, draw=False)

        for obj in objects:
            img = cvzone.overlayPNG(img, obj.img, obj.pos)
            obj.pos[1] += obj.speed

            # Check if the object goes out of bounds
            if obj.pos[1] > 630:
                if obj.isEatable:  # Only deduct health if the object is eatable
                    health -= 1  # Deduct health
                    bomb_eaten_sfx.play()  # Play sound indicating a missed eatable

                    # Check if health reaches zero and exit the game
                    if health <= 0:
                        gameOver = True  # Trigger game over state
                        break  # Exit the loop and let the game handle the transition

                # Reset the object regardless of type
                objects[objects.index(obj)] = resetObject(speed)

            if faces:
                for face in faces:
                    up, down = face[idList[0]], face[idList[1]]
                    cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2
                    upDown, _ = detector.findDistance(up, down)
                    leftRight, _ = detector.findDistance(face[idList[2]], face[idList[3]])
                    ratio = int((upDown / leftRight) * 100)
                    dist, _ = detector.findDistance((cx, cy), (obj.pos[0] + 25, obj.pos[1] + 25))

                    # Display emoji based on mouth ratio
                    emoji = open_mouth_emoji if ratio > 60 else closed_mouth_emoji
                    emoji_pos = (1150, 50)
                    img = cvzone.overlayPNG(img, emoji, emoji_pos)

                    # Check for object interaction
                    if dist < 50 and ratio > 60:
                        if obj.isEatable:
                            fruit_eaten_sfx.play()  # Play eatable sound
                            count += 1
                            objects[objects.index(obj)] = resetObject(speed)

                            # Increase speed and number of objects at intervals
                            if count % 10 == 0 and numObjects < maxObjects:
                                numObjects += 1
                                objects.append(resetObject(speed))
                                speed += 2

                                # Play the speed increase sound
                                speed_increase_sfx.play()
                        else:
                            bomb_eaten_sfx.play()  # Play bomb sound
                            health -= 1
                            objects[objects.index(obj)] = resetObject(speed)
                            if health <= 0:
                                gameOver = True

        # Display the score
        img = overlay_top_centered_text_with_pixel_font(
            img,
            f"{count} Pts",
            pixel_font,
            (255, 255, 255),
            vertical_offset=30
        )

        # Display health image
        if 0 <= health <= maxHealth:
            heart_size = (300, 200)
            resized_health_img = cv2.resize(health_images[health], heart_size, interpolation=cv2.INTER_AREA)
            img = cvzone.overlayPNG(img, resized_health_img, (50, -10))

        # Display speed image
        speed_image = speed_images.get(speed, None)
        if speed_image is not None:
            resized_speed_img = cv2.resize(speed_image, (200, 100), interpolation=cv2.INTER_AREA)
            img = cvzone.overlayPNG(img, resized_speed_img, (70, 180))
        else:
            cv2.putText(img, f"Speed: {speed}", (50, 120), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)
    else:
        # Game Over logic
        if health <= 0:
            result = gameOverScreen(count)
            if result == 'restart':
                homeScreen()
                health, speed, count = maxHealth, initialSpeed, 0
                numObjects, objects = 1, [resetObject(speed) for _ in range(numObjects)]
                gameOver = False
            elif result == 'exit':
                break

    # Check for 'r' key press to restart the game during gameplay
    key = cv2.waitKey(1)
    if key == ord('r'):
        homeScreen()
        health, speed, count = maxHealth, initialSpeed, 0
        numObjects, objects = 1, [resetObject(speed) for _ in range(numObjects)]
        gameOver = False

    elif key == ord('r'):  # Restart the game
        homeScreen()
        health, speed, count = maxHealth, initialSpeed, 0
        numObjects, objects = 1, [resetObject(speed) for _ in range(numObjects)]
        gameOver = False

    elif key == 27:  # ESC key to exit
        break

    # Show the frame
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

