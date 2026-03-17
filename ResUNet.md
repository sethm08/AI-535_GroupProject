# These are notes covering the steps taken to develop a ResUNet model.

1. Create datasets:
    - Import images
    - Import centroids
    - Split train & test subsets
2. Create a mask to interpolate the centroid values to boundary boxes (weak labeling)
    - How to best store and visualize/validate the masks?
    - Centroid diameters
