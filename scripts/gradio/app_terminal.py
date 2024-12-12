import os
import time
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from gradio_text2video import online_t2v_inference


# Constants and directories
ProjectDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CheckpointsDir = os.path.join(ProjectDir, "checkpoints")
max_image_edge = 960

def download_model():
    if not os.path.exists(CheckpointsDir):
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseV",
            local_dir=CheckpointsDir,
            max_workers=8,
        )
        toc = time.time()
        print(f"download cost {toc-tic} seconds")
    else:
        print("Already download the model.")

def limit_shape(image, input_w, input_h, img_edge_ratio, max_image_edge=max_image_edge):
    if input_h == -1 and input_w == -1:
        if isinstance(image, np.ndarray):
            input_h, input_w, _ = image.shape
        elif isinstance(image, Image.Image):
            input_w, input_h = image.size
        else:
            raise ValueError(f"image should be in [image, ndarray], but given {type(image)}")
    if img_edge_ratio == 0:
        img_edge_ratio = 1
    img_edge_ratio_infact = min(max_image_edge / max(input_h, input_w), img_edge_ratio)
    if img_edge_ratio != 1:
        return (img_edge_ratio_infact, input_w * img_edge_ratio_infact, input_h * img_edge_ratio_infact)
    else:
        return img_edge_ratio_infact, -1, -1

def limit_length(length):
    if length > 24 * 6:
        print("Length need to smaller than 144, due to gpu memory limit")
        length = 24 * 6
    return length

def main():
    parser = argparse.ArgumentParser(description='Text to Video Generation')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--image', type=str, required=True, help='Path to the reference image')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generation')
    parser.add_argument('--fps', type=int, default=6, help='Frames per second for the generated video')
    parser.add_argument('--width', type=int, default=-1, help='Width of the output video')
    parser.add_argument('--height', type=int, default=-1, help='Height of the output video')
    parser.add_argument('--video_length', type=int, default=12, help='Length of the generated video')
    parser.add_argument('--img_edge_ratio', type=float, default=1.0, help='Image edge ratio')

    args = parser.parse_args()

    download_model()

    # Load the reference image
    image = Image.open(args.image)
    image_np = np.array(image)

    # Adjust the image shape and video length
    img_edge_ratio_infact, out_w, out_h = limit_shape(image_np, args.width, args.height, args.img_edge_ratio)
    video_length = limit_length(args.video_length)

    # Generate the video
    output_video = online_t2v_inference(
        args.prompt,
        image_np,
        args.seed,
        args.fps,
        args.width,
        args.height,
        video_length,
        img_edge_ratio_infact
    )

    # Save the output video (example: save as 'output.mp4')
    #output_video.save('output.mp4')

if __name__ == "__main__":
    main()

"""
python app_terminal.py --prompt "(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)" --image "/home/user/app/MuseV/scripts/gradio/musevtests/man.png" --seed -1 --fps 6 --width -1 --height -1 --video_length 12 --img_edge_ratio 1.0

"""
