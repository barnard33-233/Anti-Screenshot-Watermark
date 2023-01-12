import imageio.v2 as imageio

img_paths = []
for i in range(1000,1001):
    img_paths.append("images/CNNplus/"+str(i)+".jpg")
    img_paths.append("images/CNNminus/"+str(i)+".jpg")
gif_images = []
for path in img_paths:
    gif_images.append(imageio.imread(path))
imageio.mimsave("test40Hz.gif", gif_images, duration=0.025)
