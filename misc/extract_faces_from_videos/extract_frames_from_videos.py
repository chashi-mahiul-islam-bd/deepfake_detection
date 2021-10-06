import cv2

def extract_frames(video_path, output_folder, save_every=10):
    """
    Extract frames from a video with given interval

    Parameters
    ----------
    video_path : str
        video path.
    output_folder : str
        output folder to save the frames.
    save_every : int, optional
        The jump between 1 famre to another. The default is 10.

    Returns
    -------
    None.

    """
    video_name = video_path # or any other extension like .avi etc
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    count = 0
    file_name = video_path.split("/")[-1].split(".")[0]
    output_general_name = output_folder + "/" + file_name + "_frame"
    while success:
        if count % save_every == 0:
            cv2.imwrite(output_general_name + "_{}.jpg".format(count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

if __name__ == "__main__":
    extract_frames("data/000_003.mp4", "data/frames")