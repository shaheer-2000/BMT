from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

import json
import subprocess
import shlex
from pathlib import Path

from pytube import YouTube, extract

import nest_asyncio
from pyngrok import ngrok
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ROOT_DIR = Path("/content/drive/MyDrive/FYP-2/inference")
VIDEOS_OUT_DIR = ROOT_DIR / "saved_videos"
FEATURES_OUT_DIR = ROOT_DIR / "saved_features"
CAPTIONS_OUT_DIR = ROOT_DIR / "saved_captions"

def download_video(video_url: str, video_id: str, output_path: Path):
	yt = YouTube(video_url)
	filtered_videos = yt.streams.filter(progressive=True, file_extension="mp4")
	if (len(filtered_videos) <= 0):
		raise Exception("Videos with extensions other than .mp4 are not supported")
	
	# low_res_videos = filtered_videos.filter(fps="30fps", resolution="480p")
	# if (len(low_res_videos) < 0):
	# 	raise Exception("Resolutions greater than 480p and FPS greater than 30fps not supported")

	try:
		filtered_videos.order_by("resolution").asc().first().download(output_path=output_path.as_posix(), filename=f"{video_id}.mp4", max_retries=2)
	except:
		raise Exception("Failed to download target video within maximum number of retries")

def extract_features(feature_type, video_path: Path, output_path: Path):
	cmd = shlex.split(f"/usr/local/envs/{feature_type}/bin/python main.py --feature_type {feature_type} --video_paths {video_path.as_posix()} --output_path {output_path.as_posix()} --extraction_fps 25 --on_extraction save_numpy --device_ids 0")
	result = subprocess.run(cmd, cwd="/content/BMT/submodules/video_features", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	print(result.stdout.decode("utf-8"))

def which_ffprobe() -> str:
    '''Determines the path to ffprobe library
    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffprobe'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffprobe_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffprobe_path

def get_video_duration(path):
	'''Determines the duration of the custom video
	Returns:
		float -- duration of the video in seconds'''
	cmd = f'{which_ffprobe()} -hide_banner -loglevel panic -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
	result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	video_duration = float(result.stdout.decode('utf-8').replace('\n', ''))
	print('Video Duration:', video_duration)
	return video_duration

def predict_captions(video_duration: float, i3d_features_path: Dict[str, Path], vggish_features_path: Path, output_path: Path):
	cmd = shlex.split(f"/usr/local/envs/bmt/bin/python /content/BMT/sample/single_video_prediction.py --duration_in_secs {video_duration} --rgb_features_path {i3d_features_path['rgb'].as_posix()} --flow_features_path {i3d_features_path['flow'].as_posix()} --vggish_features_path {vggish_features_path.as_posix()} --generated_captions_output_path {output_path.as_posix()}")
	result = subprocess.run(cmd, cwd="/content/BMT", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	print(result.stdout.decode("utf-8"))

def read_json(json_path: Path):
	with open(json_path, "r") as inp_file:
		return json.load(inp_file)

class Video(BaseModel):
	url: str

@app.post("/")
async def root(video: Video):
	"""
		- Get video URL
		- Download video using pytube
		- Pass the video to i3d & vggish and get features saved as .npy files
		- Get the video duration in seconds
		- Pass to single_video_prediction.py to get captions output file
		- Read and return as response the captions output file
	"""

	video_url = video.url
	try:
		video_id = extract.video_id(video_url)
	except:
		raise HTTPException(status_code=400, detail={ "success": False, "reason": "Invalid YouTube URL provided" })

	captions_file_path = CAPTIONS_OUT_DIR / f"{video_id}_captions.json"

	if captions_file_path in CAPTIONS_OUT_DIR.iterdir():
		response = read_json(captions_file_path)
		return response

	try:
		download_video(video_url, video_id, VIDEOS_OUT_DIR)
	except Exception as e:
		raise HTTPException(status_code=400, detail={ "success": False, "reason": e.__str__() })
	
	video_path = VIDEOS_OUT_DIR / f"{video_id}.mp4"

	extract_features("i3d", video_path, FEATURES_OUT_DIR)
	extract_features("vggish", video_path, FEATURES_OUT_DIR)

	video_duration = get_video_duration(video_path)

	i3d_features_path = {
		"rgb": FEATURES_OUT_DIR / f"{video_id}_rgb.npy",
		"flow": FEATURES_OUT_DIR / f"{video_id}_flow.npy"
	}

	predict_captions(video_duration, i3d_features_path, FEATURES_OUT_DIR / f"{video_id}_vggish.npy", captions_file_path)
	response = read_json(captions_file_path)

	return response

"""
	RUN THE APP
"""
ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

nest_asyncio.apply()
uvicorn.run(app, port=8000)