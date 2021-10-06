import face_recognition as fr
from matplotlib import pyplot as plt
import click
def extract_face(img_path):
    """
    Extract face from a given image

    Parameters
    ----------
    img_path : str
        image path.

    Returns
    -------
    None.

    """
    image = fr.load_image_file(img_path)
    face_locations = fr.face_locations(image)
    face = image[ face_locations[0][0]:face_locations[0][2],face_locations[0][3]:face_locations[0][1], :]
    plt.imshow(face, interpolation='nearest')
    plt.show()

@click.command()
@click.option("--img_path", default="data/frames/000_003frame_0.jpg", help="Image path")
@click.option("--task", default="extract_face", help="Type of task: extract_face")
def main(img_path, task):
    """
    Main program to run all the functions individually

    Parameters
    ----------
    img_path : str
        image path.
    task : str
        type of the task.

    Returns
    -------
    None.

    """
    if task == "extract_face":
        extract_face(img_path)
        
if __name__ == "__main__":
    main()