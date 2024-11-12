with open('Scene000000_CAM® NonTnrFrm_D5_FI_W 0 Preview P501 LME _Emt-GREY _Comp-0_PCnt-2 Meta-0_L1008x756_20241007', 'rb') as fd:
    im_invec = np.fromfile(fd, dtype=np.uint8, count=656*496)
im1 = np.reshape(im_invec, (656, 496))

# Save the original image
Image.fromarray(im1).save('original_image.png')

# Read the uv data
with open('Scene000000_CAMO NonTnrFrm D5 F1 W 2 Preview P591 LME DST_Fat-GREY_Comp-0_PCnt-2 Meta-0_504x189_20241007_144127044.raw', 'rb') as fd:
    uvdata = np.fromfile(fd, dtype=np.int16, count=168*124)

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
siz = 3
xgrid = np.arange(0, 124, siz)
ygrid = np.arange(0, 82, siz)
xi, yi = np.meshgrid(xgrid, ygrid)

# Create interpolation functions
interp_u = RectBivariateSpline(np.arange(u_cropped.shape[0]), np.arange(u_cropped.shape[1]), u_cropped)
interp_v = RectBivariateSpline(np.arange(v_cropped.shape[0]), np.arange(v_cropped.shape[1]), v_cropped)

# Interpolate u and v
Vxi = interp_u(yi[:,0], xi[0,:])
Vyi = interp_v(yi[:,0], xi[0,:])

# Plot and save the image with quiver
plt.figure()
plt.imshow(im1_scaled, cmap='gray')
plt.quiver(xi, yi, Vxi, Vyi, scale=3, color='red')
plt.axis('image')
plt.savefig('image_with_quiver.png')
plt.close()
