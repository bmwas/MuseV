import os
import time
import argparse
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from gradio_text2video import online_t2v_inference
from huggingface_hub import HfApi, HfFolder, Repository, create_repo
import logging
from dotenv import load_dotenv
load_dotenv()
# Constants and directories
ProjectDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CheckpointsDir = os.path.join(ProjectDir, "checkpoints")
max_image_edge = 960


def upload_files_to_hf(repo_id, video_path, image_path, target_dir='', token=None):
    """
    Uploads the specified video and image files to the HuggingFace repository.

    Args:
        repo_id (str): The repository ID in the format 'username/repo_name'.
        video_path (str): The local path to the video file to upload.
        image_path (str): The local path to the image file to upload.
        target_dir (str, optional): The target directory in the repo to upload the files to.
                                    Defaults to the root directory.
        token (str, optional): The HuggingFace API token. If not provided, it will be read from the
                               environment variable 'HUGGINGFACE_TOKEN'.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    api = HfApi()

    # Retrieve the token
    if token is None:
        token = os.getenv('HUGGINGFACE_TOKEN')
        if token is None:
            raise ValueError("No token provided and 'HUGGINGFACE_TOKEN' not set.")

    # Check if the repository exists; if not, create it
    try:
        api.repo_info(repo_id, token=token)
        logger.info(f"Repository '{repo_id}' found.")
    except Exception as e:
        logger.warning(f"Repository '{repo_id}' not found. Attempting to create it.")
        try:
            create_repo(repo_id=repo_id, token=token)
            logger.info(f"Repository '{repo_id}' created successfully.")
        except Exception as create_e:
            logger.error(f"Failed to create repository '{repo_id}': {create_e}")
            raise create_e

    # Determine the target paths in the repo
    video_filename = os.path.basename(video_path)
    image_filename = os.path.basename(image_path)
    if target_dir:
        video_target = os.path.join(target_dir, video_filename)
        image_target = os.path.join(target_dir, image_filename)
    else:
        video_target = video_filename
        image_target = image_filename

    # Upload the video
    try:
        logger.info(f"Uploading {video_path} to {repo_id}/{video_target}...")
        api.upload_file(
            path_or_fileobj=video_path,
            path_in_repo=video_target,
            repo_id=repo_id,
            repo_type='model',
            token=token,
        )
        logger.info(f"Video '{video_filename}' uploaded successfully.")
    except Exception as upload_video_e:
        logger.error(f"Failed to upload video '{video_filename}': {upload_video_e}")
        raise upload_video_e

    # Upload the image
    try:
        logger.info(f"Uploading {image_path} to {repo_id}/{image_target}...")
        api.upload_file(
            path_or_fileobj=image_path,
            path_in_repo=image_target,
            repo_id=repo_id,
            repo_type='model',
            token=token,
        )
        logger.info(f"Image '{image_filename}' uploaded successfully.")
    except Exception as upload_image_e:
        logger.error(f"Failed to upload image '{image_filename}': {upload_image_e}")
        raise upload_image_e

    logger.info("Both files uploaded successfully.")

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
    parser.add_argument('--hf_repo_id', type=str, required=True, help='HuggingFace repository ID (e.g., "username/repo_name")')
    parser.add_argument('--hf_target_dir', type=str, default='', help='Target directory in the repo to upload files to')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token (optional if set as env variable)')
    

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
    # Upload the video and image to HuggingFace Hub
    try:
        upload_files_to_hf(
            repo_id=args.hf_repo_id,
            video_path=args.hf_target_dir,
            image_path=args.image,
            target_dir=args.hf_target_dir,
            token=args.hf_token
        )
    except Exception as e:
        print(f"Failed to upload files to HuggingFace Hub: {e}")
    else:
        print("Files uploaded to HuggingFace Hub successfully.")

if __name__ == "__main__":
    main()

"""
python app_terminal.py --prompt "(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)" --image "/home/user/app/MuseV/scripts/gradio/musevtests/man.png" --seed -1 --fps 6 --width -1 --height -1 --video_length 12 --img_edge_ratio 1.0


python app_terminal.py \
    --prompt "(masterpiece, best quality, highres:1),(1boy, solo:1),(eye blinks:1.8),(head wave:1.3)" \
    --image "/home/user/app/MuseV/scripts/gradio/musevtests/man.png" \
    --seed -1 \
    --fps 6 \
    --width -1 \
    --height -1 \
    --video_length 12 \
    --img_edge_ratio 1.0 \
    --hf_repo_id "Benson/musetalkmodels" \
    --hf_target_dir "/home/user/app/MuseV/scripts/gradio/results" 

"""
