import os
import shutil
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

def copy_png_files(src_directory, dst_directory):
    """
    Copies all .png files from the source directory to the destination directory.
    If the destination directory does not exist, it is created.
    
    :param src_directory: Path to the source directory containing .png files.
    :param dst_directory: Path to the destination directory where .png files will be copied.
    """
    # Ensure the destination directory exists
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
    
    # Iterate through the source directory
    for filename in os.listdir(src_directory):
        # Check if the file is a .png file
        if filename.lower().endswith('.png'):
            src_file_path = os.path.join(src_directory, filename)
            dst_file_path = os.path.join(dst_directory, filename)
            # Copy the file to the destination directory
            shutil.copy2(src_file_path, dst_file_path)

def upload_files_to_hf(repo_id, video_path, image_path, target_dir='', token=None):
    """
    Uploads the specified video and image files (or directories) to the HuggingFace repository.
    If a directory is provided instead of a file, all files within the directory are uploaded.

    Args:
        repo_id (str): The repository ID in the format 'username/repo_name'.
        video_path (str): The local path to the video file or directory to upload.
        image_path (str): The local path to the image file or directory to upload.
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

    def upload_single_file(local_file_path, repo_id, repo_target_path):
        # Upload a single file
        logger.info(f"Uploading {local_file_path} to {repo_id}/{repo_target_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_file_path,
                path_in_repo=repo_target_path,
                repo_id=repo_id,
                repo_type='model',
                token=token,
            )
            logger.info(f"File '{local_file_path}' uploaded successfully.")
        except Exception as e:
            logger.error(f"Failed to upload file '{local_file_path}': {e}")
            raise e

    def upload_path(local_path, repo_id, base_target_dir=''):
        # Check if local_path is a directory or a file
        if os.path.isdir(local_path):
            # Recursively upload all files in the directory
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_local_path = os.path.join(root, file)
                    # Determine relative path to maintain directory structure in repo
                    relative_path = os.path.relpath(file_local_path, local_path)
                    repo_target_path = os.path.join(base_target_dir, relative_path)
                    upload_single_file(file_local_path, repo_id, repo_target_path)
        else:
            # It's a single file
            filename = os.path.basename(local_path)
            repo_target_path = os.path.join(base_target_dir, filename) if base_target_dir else filename
            upload_single_file(local_path, repo_id, repo_target_path)

    # Upload the "video_path" (file or directory)
    upload_path(video_path, repo_id, target_dir)

    # Upload the "image_path" (file or directory)
    #upload_path(image_path, repo_id, target_dir)

    logger.info("All specified files/directories uploaded successfully.")

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
    parser.add_argument('--hf_video_path', type=str, default='', help='Path to video in local directory')
    parser.add_argument('--hf_image_path', type=str, default='', help='Path to image (i.e.image used to seed video generation)')
    

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
            video_path=args.hf_video_path,
            image_path=args.hf_image_path,
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
    --hf_video_path "/home/user/app/MuseV/scripts/gradio/results" \
    --hf_image_path "/home/user/app/MuseV/scripts/gradio/musevtests"

"""
