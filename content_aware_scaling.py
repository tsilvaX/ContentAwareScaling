import cv2
import numpy as np

def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Simple gradient magnitude as energy
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobelx) + np.abs(sobely)
    return energy

def find_vertical_seam(energy):
    rows, cols = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, rows):
        for j in range(cols):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                M[i, j] += M[i-1, idx+j]
            else:
                idx = np.argmin(M[i-1, j-1:min(j+2, cols)])
                backtrack[i, j] = idx + j - 1
                M[i, j] += M[i-1, idx+j-1]
    # Find end of the minimum seam
    seam = np.zeros(rows, dtype=int)
    seam[-1] = np.argmin(M[-1])
    for i in range(rows-2, -1, -1):
        seam[i] = backtrack[i+1, seam[i+1]]
    return seam

def remove_vertical_seam(img, seam):
    rows, cols, _ = img.shape
    output = np.zeros((rows, cols-1, 3), dtype=np.uint8)
    for i in range(rows):
        j = seam[i]
        output[i, :, 0] = np.delete(img[i, :, 0], j)
        output[i, :, 1] = np.delete(img[i, :, 1], j)
        output[i, :, 2] = np.delete(img[i, :, 2], j)
    return output

def content_aware_scale(img, num_seams):
    for _ in range(num_seams):
        energy = compute_energy(img)
        seam = find_vertical_seam(energy)
        img = remove_vertical_seam(img, seam)
    return img

if __name__ == "__main__":
    input_file = "input.jpg"  # Put your image here
    output_file = "output.jpg"
    num_seams_to_remove = 5  # Change this number to remove more/less

    img = cv2.imread(input_file)
    if img is None:
        print("Error: Could not read the image file.")
        exit()

    result = content_aware_scale(img, num_seams_to_remove)
    cv2.imwrite(output_file, result)
    print(f"Saved scaled image as {output_file}")