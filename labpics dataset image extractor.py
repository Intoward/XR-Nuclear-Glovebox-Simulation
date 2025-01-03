import os
import shutil

def copy_and_rename_images(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Initialize a counter for renaming the images
    counter = 1

    # Walk through the source folder
    for root, dirs, files in os.walk(source_folder):
        # Check if 'Image.jpg' exists in the current directory
        if 'Image.jpg' in files:
            source_path = os.path.join(root, 'Image.jpg')
            destination_path = os.path.join(destination_folder, f"{counter}.jpg")

            # Copy and rename the file
            shutil.copy(source_path, destination_path)
            print(f"Copied and renamed: {source_path} -> {destination_path}")

            # Increment the counter
            counter += 1

if __name__ == "__main__":
    # Define the source folder (e.g., 'Test') and destination folder (e.g., 'images')
    source_folder = input("Enter the path to the source folder: ").strip()
    destination_folder = os.path.join(os.getcwd(), "images")

    # Run the function
    copy_and_rename_images(source_folder, destination_folder)
    print(f"All images have been copied and renamed into the folder: {destination_folder}")
