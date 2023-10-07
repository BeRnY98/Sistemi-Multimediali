import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

progetto = "C:\\Users\\Family\\Desktop\\BERNY\\università\\Sistemi Multimediali\\PROGETTO"
datasetsDirectory = os.path.join("images", "modificati")
if not os.path.exists(datasetsDirectory):
        os.makedirs(datasetsDirectory)
gta_files = os.listdir("images\\")

for file in tqdm(gta_files):
    if file == "modificati":
        continue

    # The next step is to read the image using the imread function
    # and then convert it to RGB format with the help of the cvtColor function. 
    # We then plot the image using the imshow function.
    path = os.path.join("images", file)
    img = cv2.imread(path) # Imread = Lettura immagine
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # ctvColor = Conversione da BGR a RGB

    # The next step in the process is to convert the image into a grayscale format using the cvtColor function. 
    # The reason behind doing so is that it simplifies the process and helps in the time complexity of the program later on.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # To make things simpler for us, we will get an edged image of the grayscale image and then apply the convolutional network to the image.
    # The same is done by using the adaptiveThreshold and setting the required parameters to get the edged image. The code for the same is displayed below.
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    """
    plt.figure(figsize=(10,10))
    plt.imshow(edges,cmap="gray")
    plt.axis("off")
    plt.title("Cartoonized Image")
    plt.show()
    """
    cv2.imwrite(os.path.join(datasetsDirectory, file), edges)

print("Done.")
