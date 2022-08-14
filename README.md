# Document-Scanner

The algorithm takes an input image (document/page), it is converted into grayscale, edge of the document is found out using canny edge detector, then corner points are extracted from the images, then hough lines and perspective transformation are applied to get the top down shape of the image. Finally adative thresholding is applied to get the binary image.
