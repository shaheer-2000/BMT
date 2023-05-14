import subprocess

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
    cmd = f'{which_ffprobe()} -hide_banner -loglevel panic' \
          f' -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path}'
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    video_duration = float(result.stdout.decode('utf-8').replace('\n', ''))
    print('Video Duration:', video_duration)
    return video_duration