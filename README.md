bits_per_line = width * 10  # 10 bits per pixel
    bytes_per_line = bits_per_line // 8
    padding_bytes_per_line = 24  # extra bytes per line

    with open(input_filename, "rb") as f:
        img_data = []
        for line in range(height):
            line_data = f.read(bytes_per_line)
            byte_array = np.frombuffer(line_data, dtype=np.uint8)
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

            pixels_line = np.column_stack((pixel0, pixel1, pixel2, pixel3))
            img_data.append(pixels_line)

            # Skip the padding bytes
            f.read(padding_bytes_per_line)

    image_array = np.vstack(img_data)
    raw_image = image_array.reshape((height, width))

    # Scale the 10-bit raw image data to 8-bit
    raw_image_8bit = (raw_image >> 2).astype(np.uint8)

    # Create the Bayer output image
    bayer_output = np.zeros((height, width, 3), dtype=np.uint8)

    # Map the raw image data to the correct color channels
    bayer_output[0::2, 0::2, 1] = raw_image_8bit[0::2, 0::2]  # Gr (Green on red line)
    bayer_output[0::2, 1::2, 0] = raw_image_8bit[0::2, 1::2]  # R (Red)
    bayer_output[1::2, 0::2, 2] = raw_image_8bit[1::2, 0::2]  # B (Blue)
    bayer_output[1::2, 1::2, 1] = raw_image_8bit[1::2, 1::2]  # Gb (Green on blue line)
