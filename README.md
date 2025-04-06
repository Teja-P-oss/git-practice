def floyd_steinberg_dither(self, input_bit_depth=10, output_bit_depth=6):
        """
        Floyd-Steinberg dithering with single line buffer for error diffusion
        Reducing from input_bit_depth to output_bit_depth (e.g., 10-bit to 6-bit)
        """
        input_max = (1 << input_bit_depth) - 1  # 1023 for 10-bit
        output_max = (1 << output_bit_depth) - 1  # 63 for 6-bit

        # Calculate the bit shift amount
        bit_shift = input_bit_depth - output_bit_depth  # 4 bits

        # Create a deep copy of the image to work with
        result = self.image.copy().astype(np.float32)

        # Single-line error buffer (for bottom pixels)
        error_buffer = np.zeros((self.width, 3), dtype=np.float32)

        for y in range(self.height):
            # Reset error buffer at the start of each new row
            if y > 0:
                error_buffer = np.zeros((self.width, 3), dtype=np.float32)

            for x in range(self.width):
                for c in range(3):
                    # Add error from buffer to current pixel
                    old_pixel = result[y, x, c] + error_buffer[x, c]
                    old_pixel = np.clip(old_pixel, 0, input_max)

                    # Perform bit reduction by quantizing
                    # For 10->6 bit, we're keeping the 6 most significant bits
                    new_pixel_reduced = np.floor(old_pixel / (1 << bit_shift))

                    # Calculate quantization error
                    quant_error = old_pixel - (new_pixel_reduced * (1 << bit_shift))

                    # Save the quantized value (scaled to 0-output_max range)
                    result[y, x, c] = np.clip(new_pixel_reduced, 0, output_max)

                    # Distribute error to neighboring pixels using Floyd-Steinberg weights
                    if x < self.width - 1:
                        error_buffer[x + 1, c] += quant_error * 7/16  # right

                    if y < self.height - 1:  # Use error buffer for next row
                        if x > 0:
                            error_buffer[x - 1, c] += quant_error * 3/16  # bottom-left
                        error_buffer[x, c] += quant_error * 5/16  # bottom
                        if x < self.width - 1:
                            error_buffer[x + 1, c] += quant_error * 1/16  # bottom-right

        # Scale the result back to the original range (approximately) for internal use if needed
        self.image = (result * (input_max / output_max)).astype(np.uint16)
        return self.image
