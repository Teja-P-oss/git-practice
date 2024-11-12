import numpy as np

def bayerp(input_path):
    import numpy as np

    height = 3000
    width = 4000
    input_filename = input_path
    output_filename = "C:/Users/teia-potti-SESDS/Desktop/camera/scripts/Bayerp.bmp"

    bits_per_line = width * 10  # 10 bits per pixel
    bytes_per_line = bits_per_line // 8  # Total bytes per line for pixel data
    padding_bytes_per_line = 24  # Extra bytes per line
    total_bytes_per_line = bytes_per_line + padding_bytes_per_line  # Total bytes per line including padding

    # Read the entire file into a numpy array
    with open(input_filename, "rb") as f:
        data = f.read()

    # Ensure the data length matches expected size
    expected_size = total_bytes_per_line * height
    if len(data) != expected_size:
        raise ValueError(f"Unexpected file size: expected {expected_size}, got {len(data)}")

    # Convert data to numpy array
    data = np.frombuffer(data, dtype=np.uint8)
    data = data.reshape(height, total_bytes_per_line)

    # Extract pixel data by excluding padding bytes
    pixel_data = data[:, :bytes_per_line].flatten()

    # Ensure the pixel data length is a multiple of 5 bytes (since 5 bytes -> 4 pixels)
    if len(pixel_data) % 5 != 0:
        raise ValueError("Pixel data length is not a multiple of 5 bytes.")

    # Reshape pixel data to process 5-byte blocks
    byte_array = pixel_data.reshape(-1, 5)

    # Unpack the 10-bit pixels from 5-byte blocks
    b0 = byte_array[:, 0].astype(np.uint16)
    b1 = byte_array[:, 1].astype(np.uint16)
    b2 = byte_array[:, 2].astype(np.uint16)
    b3 = byte_array[:, 3].astype(np.uint16)
    b4 = byte_array[:, 4].astype(np.uint16)

    pixel0 = ((b0 << 2) | (b1 >> 6)) & 0x3FF
    pixel1 = (((b1 & 0x3F) << 4) | (b2 >> 4)) & 0x3FF
    pixel2 = (((b2 & 0x0F) << 6) | (b3 >> 2)) & 0x3FF
    pixel3 = (((b3 & 0x03) << 8) | b4) & 0x3FF

    # Combine pixels and reshape to image dimensions
    pixels = np.column_stack((pixel0, pixel1, pixel2, pixel3)).flatten()
    raw_image = pixels.reshape(height, width)

    # Create Bayer output
    bayer_output = np.zeros((height, width, 3), dtype=np.uint16)
    bayer_output[0::2, 0::2, 0] = raw_image[0::2, 0::2]  # Red channel
    bayer_output[0::2, 1::2, 1] = raw_image[0::2, 1::2]  # Green channel
    bayer_output[1::2, 0::2, 1] = raw_image[1::2, 0::2]  # Green channel
    bayer_output[1::2, 1::2, 2] = raw_image[1::2, 1::2]  # Blue channel

    # Optionally, normalize the data for display or saving
    bayer_output = (bayer_output / 4).astype(np.uint8)  # Scale 10-bit data to 8-bit

    # Save or display the image as needed
    # For example, using OpenCV to save the image:
    # import cv2
    # cv2.imwrite(output_filename, bayer_output)

    return bayer_output
