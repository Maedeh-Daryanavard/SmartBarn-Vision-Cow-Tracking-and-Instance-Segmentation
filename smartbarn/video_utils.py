from pathlib import Path
import subprocess

def to_h264(input_path,output_path=None):
    """Convert mp4v to browser-compatible H.264 without rerunning inference."""
    import imageio_ffmpeg
    src=Path(input_path)
    dst=Path(output_path) if output_path else src.with_name(src.stem+'_h264.mp4')
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(src),'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(dst)]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    return dst
