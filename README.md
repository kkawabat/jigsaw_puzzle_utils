# Jigsaw Puzzle Utils

A helper python library that uses computer vision to help solve jigsaw puzzle

## context

I recently went to Japan and got a pretty [great picture](docs/images/original_img.png) of Tokyo cityscape from Tokyo Skytree, the tallest building in Japan.   

As kind of spur of the moment, I decided to order a [jigsaw puzzle](docs/images/puzzle_box.jpg) of the photo.  

However, after trying to assemble it for a few hours I realized this was more difficult than I imagine and thought this might make for an opportunity to make this into a fun computer vision project.

## installation

- install python 3.11 + poetry
- run `poetry install`


## notes & insights
- things that messes up the contour detection
  - touching pieces
  - noisy background (ended up buying a green board)
  - image on the pieces (flip over to the back of the pieces so that the pieces are blanks)
  - shadow of the pieces (having an angled light source adds shadows on the opposite side)
  - distance/angle of the image (further pieces are smaller, angling distorts the shape, you need to account for that)


## additional references
- initial video i saw [link](https://www.youtube.com/watch?v=WsPHBD5NsS0)
- found this medium article [link](https://medium.com/data-science/solving-jigsaw-puzzles-with-python-and-opencv-d775ba730660)
- Interesting video that uses peak detector on distance/angle from center to find the corners [link](https://www.youtube.com/watch?v=Bo0RSxt5ECI&ab_channel=RujutaVaze)