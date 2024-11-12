with open(path1, 'rb') as fd:
    buffer = fd.read(656 * 496 * np.dtype(np.uint8).itemsize)
im_invec = np.frombuffer(buffer, dtype=np.uint8, count=656*496)
im1 = np.reshape(im_invec, (656, 496))

# Save the original image
Image.fromarray(im1).save('original_image.png')

# Read the uv data using path2
with open(path2, 'rb') as fd:
    buffer = fd.read(168 * 124 * np.dtype(np.int16).itemsize)
uvdata = np.frombuffer(buffer, dtype=np.int16, count=168*124)

# Separate u and v components
uvec = uvdata[::2]  # Take every other sample starting from index 0
vvec = uvdata[1::2]  # Take every other sample starting from index 1

# Reshape u and v vectors
u = uvec.reshape((84, 124))
v = vvec.reshape((84, 124))

# Crop u and v to match the image size
u_cropped = u[:82, :]
v_cropped = v[:82, :]

# Resize the image to match u and v dimensions
im1_scaled = cv2.resize(im1, (124, 82), interpolation=cv2.INTER_AREA)

# Save the scaled image
Image.fromarray(im1_scaled).save('scaled_image.png')

# Interpolate u and v to match the grid
siz = 5  # Adjusted for fewer arrows
xgrid = np.arange(0, 124, siz)
ygrid = np.arange(0, 82, siz)
xi, yi = np.meshgrid(xgrid, ygrid)

# Create interpolation functions
interp_u = RectBivariateSpline(np.arange(u_cropped.shape[0]), np.arange(u_cropped.shape[1]), u_cropped)
interp_v = RectBivariateSpline(np.arange(v_cropped.shape[0]), np.arange(v_cropped.shape[1]), v_cropped)

# Interpolate u and v
Vxi = interp_u(yi[:,0], xi[0,:])
Vyi = interp_v(yi[:,0], xi[0,:])

# Scale down the vector components to reduce arrow length
scaling_factor = 20  # Adjust as needed
Vxi_scaled = Vxi / scaling_factor
Vyi_scaled = Vyi / scaling_factor

# Compute the magnitude of vectors
magnitude = np.hypot(Vxi_scaled, Vyi_scaled)

# Cap the maximum arrow length
max_arrow_length = 3  # Adjust as needed
mask = magnitude > max_arrow_length
Vxi_scaled[mask] = Vxi_scaled[mask] * max_arrow_length / magnitude[mask]
Vyi_scaled[mask] = Vyi_scaled[mask] * max_arrow_length / magnitude[mask]

# Plot and save the image with quiver
plt.figure(figsize=(8, 6))
plt.imshow(im1_scaled, cmap='gray')
plt.quiver(xi, yi, Vxi_scaled, Vyi_scaled, color='red', scale_units='xy', angles='xy', scale=1)
plt.axis('image')
plt.savefig('image_with_quiver.png')
plt.close()
