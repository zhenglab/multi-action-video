import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed

def audio_process(video_file_path, dst_root_path, ext):
    if ext != video_file_path.suffix:
        return
    
    ffprobe_cmd = ('ffprobe -v error -show_entries '
                   'stream=codec_type -of csv=p=0').split()
    ffprobe_cmd.append(str(video_file_path))

    p = subprocess.run(ffprobe_cmd, capture_output=True)
    res = p.stdout.decode('utf-8').splitlines()
    
#     print(res)
    if 'audio' in res:
        name = video_file_path.stem + ".wav"
        dst_file_path = dst_root_path / name
    
        ffmpeg_cmd = ['ffmpeg', '-loglevel', 'panic', '-i', str(video_file_path)]
        ffmpeg_cmd += ['-q:a', '0', '-map', 'a']
        ffmpeg_cmd += ['-threads', '1', '{}'.format(dst_file_path)]
        
        print(ffmpeg_cmd)
        subprocess.run(ffmpeg_cmd)
        print('\n')

def class_process(class_dir_path, dst_root_path, ext, fps=-1, size=240):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        audio_process(video_file_path, dst_class_path, ext)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of audios')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    args = parser.parse_args()

    ext = '.mp4'
    class_dir_paths = [x for x in sorted(args.dir_path.iterdir())]
    test_set_video_path = args.dir_path / 'test'
    if test_set_video_path.exists():
        class_dir_paths.append(test_set_video_path)

    status_list = Parallel(
        n_jobs=args.n_jobs,
        backend='threading')(delayed(class_process)(
            class_dir_path, args.dst_path, ext)
                for class_dir_path in class_dir_paths)
