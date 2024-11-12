bits_per_line = width * 10  # 10 bits per pixel
    bytes_per_line = bits_per_line // 8
    padding_bytes_per_line = 24  # Extra padding bytes per line

    with open(input_filename, "rb") as f:
        img_data = []
        for line in range(height):
            line_data = f.read(bytes_per_line)
            byte_array = np.frombuffer(line_data, dtype=np.uint8)
            byte_array = byte_array.reshape(-1, 5)
            b0 = byte_array[:, 0]
            b1 = byte_array[:, 1]
            b2 = byte_array[:, 2]
            b3 = byte_array[:, 3]
            b4 = byte_array[:, 4]

            pixel0 = ((b0 << 2) | (b1 >> 6)) & 0x3FF
            pixel1 = (((b1 & 0x3F) << 4) | (b2 >> 4)) & 0x3FF
            pixel2 = (((b2 & 0x0F) << 6) | (b3 >> 2)) & 0x3FF
            pixel3 = (((b3 & 0x03) << 8) | b4) & 0x3FF

            pixels_line = np.column_stack((pixel0, pixel1, pixel2, pixel3))
            img_data.append(pixels_line)
            f.read(padding_bytes_per_line)  # Skip padding bytes

    image_array = np.vstack(img_data)
    raw_image = image_array.reshape((height, width))

    # Scale 10-bit data to 8-bit
    raw_image = (raw_image >> 2).astype(np.uint8)

    # Initialize Bayer output image
    bayer_output = np.zeros((height, width, 3), dtype=np.uint8)

    # Correctly map the Bayer pattern
    bayer_output[0::2, 0::2, 1] = raw_image[0::2, 0::2]  # Green (Gr)
    bayer_output[0::2, 1::2, 0] = raw_image[0::2, 1::2]  # Red (R)
    bayer_output[1::2, 0::2, 2] = raw_image[1::2, 0::2]  # Blue (B)
    bayer_output[1::2, 1::2, 1] = raw_image[1::2, 1::2]  # Green (Gb)
