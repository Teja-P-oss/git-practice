import numpy as np

def bayerp(input_path):
    bayer_text = "Bayer_Image"
    height = 3000
    width = 4000
    input_filename = input_path
    output_filename = "C:/Users/teia-potti-SESDS/Desktop/camera/scripts/Bayerp.bmp"

    bits_per_line = width * 10  # 10 bits per pixel
    bytes_per_line = bits_per_line // 8
    if bits_per_line % 8 != 0:
        bytes_per_line += 1
    padding_bytes_per_line = 24  # extra bytes per line

    with open(input_filename, "rb") as f:
        img_data = []
        for line in range(height):
            line_data = f.read(bytes_per_line)
            byte_array = np.frombuffer(line_data, dtype=np.uint8)
            if len(byte_array) < bytes_per_line:
                # Handle incomplete line if necessary
                continue
            byte_array = byte_array.reshape(-1, 5)

            b0 = byte_array[:, 0].astype(np.uint16)
            b1 = byte_array[:, 1].astype(np.uint16)
            b2 = byte_array[:, 2].astype(np.uint16)
            b3 = byte_array[:, 3].astype(np.uint16)
            b4 = byte_array[:, 4].astype(np.uint16)

            pixel0 = ((b0 << 2) | (b1 >> 6)) & 0x3FF
            pixel1 = (((b1 & 0x3F) << 4) | (b2 >> 4)) & 0x3FF
            pixel2 = (((b2 & 0x0F) << 6) | (b3 >> 2)) & 0x3FF
            pixel3 = (((b3 & 0x03) << 8) | b4) & 0x3FF

            # Flatten the pixels and append to the list
            pixels_line = np.column_stack((pixel0, pixel1, pixel2, pixel3)).flatten()
            img_data.append(pixels_line)
            # Read and discard padding bytes
            f.read(padding_bytes_per_line)

    # Convert the list of arrays into a single NumPy array
    image_array = np.array(img_data)
    raw_image = image_array.reshape((height, width))

    # Convert 10-bit pixels to 8-bit by shifting
    raw_image_8bit = (raw_image >> 2).astype(np.uint8)

    # Create the Bayer output image
    bayer_output = np.zeros((height, width, 3), dtype=np.uint8)
    # Red channel
    bayer_output[0::2, 0::2, 0] = raw_image_8bit[0::2, 0::2]
    # Green channel (two positions)
    bayer_output[0::2, 1::2, 1] = raw_image_8bit[0::2, 1::2]
    bayer_output[1::2, 0::2, 1] = raw_image_8bit[1::2, 0::2]
    # Blue channel
    bayer_output[1::2, 1::2, 2] = raw_image_8bit[1::2, 1::2]

    # Save or display the image as needed
    # For example, using OpenCV to save the image:
    # cv2.imwrite(output_filename, bayer_output)
