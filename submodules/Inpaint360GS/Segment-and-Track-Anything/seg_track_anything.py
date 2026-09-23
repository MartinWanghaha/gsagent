import os
from pathlib import Path
import shutil
import zipfile

import cv2
from model_args import segtracker_args,sam_args,aot_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import gc
import imageio
from scipy.ndimage import binary_dilation


_TRACKER_ROOT = Path(__file__).resolve().parent


def _path_from_environment(variable_name, default):
    """Resolve an optional tracker path override without caching the environment."""
    value = os.environ.get(variable_name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def tracker_assets_root():
    """Directory used to unpack uploaded image sequences."""
    return _path_from_environment("TRACKER_ASSETS_ROOT", _TRACKER_ROOT / "assets")


def tracker_image_sequence():
    """Image-sequence archive shown as the default Gradio example."""
    return _path_from_environment(
        "TRACKER_IMAGE_SEQUENCE", _TRACKER_ROOT / "assets" / "images.zip"
    )


def tracking_results_root():
    """Root directory for tracker masks, previews, videos, and archives."""
    return _path_from_environment(
        "TRACKING_RESULTS_ROOT", _TRACKER_ROOT / "tracking_results"
    )


def _child_path(root, child_name):
    """Resolve a child path and reject attempts to leave or select its root."""
    root = Path(root).resolve()
    child = (root / child_name).resolve()
    if child == root or root not in child.parents:
        raise ValueError(f"Path must be a child of {root}: {child}")
    return child


def image_sequence_workspace(input_img_seq):
    """Return the extraction directory associated with a Gradio upload."""
    archive = Path(getattr(input_img_seq, "name", input_img_seq))
    return _child_path(tracker_assets_root(), archive.stem)


def tracking_result_workspace(video_name):
    """Return the isolated result directory for one video or image sequence."""
    return _child_path(tracking_results_root(), video_name)


def _path_within_run(path, run_root):
    """Validate an output path before mutating files in a tracker run."""
    results_root = tracking_results_root()
    run_root = Path(run_root).resolve()
    if run_root == results_root or results_root not in run_root.parents:
        raise ValueError(
            f"Tracker run must be inside {results_root}: {run_root}"
        )
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing tracker output symlink: {path}")
    resolved = path.resolve()
    if resolved == run_root or run_root not in resolved.parents:
        raise ValueError(f"Tracker output must be inside {run_root}: {resolved}")
    return resolved


def _clear_output_directory(path, run_root):
    """Remove a known per-run output directory without invoking a shell."""
    output_dir = _path_within_run(path, run_root)
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise NotADirectoryError(output_dir)


def _archive_prediction_masks(io_args, video_name):
    """Create the downloadable mask archive with a stable top-level folder."""
    run_root = tracking_result_workspace(video_name)
    configured_root = Path(io_args['tracking_result_dir']).resolve()
    if configured_root != run_root:
        raise ValueError(
            f"Unexpected tracking result directory: {configured_root} != {run_root}"
        )

    mask_dir = _path_within_run(io_args['output_mask_dir'], run_root)
    archive_path = _path_within_run(
        run_root / f"{video_name}_pred_mask.zip", run_root
    )
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Prediction mask directory not found: {mask_dir}")

    with zipfile.ZipFile(
        archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr(f"{mask_dir.name}/", b"")
        for mask_path in sorted(mask_dir.rglob("*")):
            resolved_mask = _path_within_run(mask_path, run_root)
            if resolved_mask.is_file():
                archive.write(
                    resolved_mask,
                    arcname=resolved_mask.relative_to(mask_dir.parent),
                )
    return archive_path

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

def create_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    


aot_model2ckpt = {
    "deaotb": "./ckpt/DeAOTB_PRE_YTB_DAV.pth",
    "deaotl": "./ckpt/DeAOTL_PRE_YTB_DAV",
    "r50_deaotl": "./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth",
}


def tracking_objects_in_video(SegTracker, input_video, input_img_seq, fps, frame_num=0):
    
    if input_video is not None:
        video_name = os.path.basename(input_video).split('.')[0]
    elif input_img_seq is not None:
        file_path = image_sequence_workspace(input_img_seq)
        if not file_path.is_dir():
            return None, None
        imgs_path = sorted(str(path) for path in file_path.iterdir())
        video_name = file_path.name
    else:
        return None, None

    # create dir to save result 
    tracking_result_dir = tracking_result_workspace(video_name)
    create_dir(tracking_result_dir)
    
    io_args = {
        'tracking_result_dir': str(tracking_result_dir),
        'output_mask_dir': str(tracking_result_dir / f'{video_name}_masks'),
        'output_masked_frame_dir': str(
            tracking_result_dir / f'{video_name}_masked_frames'
        ),
        'output_video': str(tracking_result_dir / f'{video_name}_seg.mp4'),
        'output_gif': str(tracking_result_dir / f'{video_name}_seg.gif'),
    }

    if input_video is not None:
        return video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num)
    elif input_img_seq is not None:
        return img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num)


def video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num=0):

    pred_list = []
    masked_pred_list = []

    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_num > 0:
        output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
        output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])

        for i in range(0, frame_num):
            cap.read()
            pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
            masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

    
    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        run_root = io_args['tracking_result_dir']
        _clear_output_directory(io_args['output_mask_dir'], run_root)
        _clear_output_directory(io_args['output_masked_frame_dir'], run_root)
    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame  = cap.read()  
            if not ret:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, str(frame_idx+frame_num).zfill(5) + '_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            save_prediction(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        cap.release()
        print('\nfinished')
    
    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(input_video)
    # if frame_num > 0:
    #     for i in range(0, frame_num):
    #         cap.read()  
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # if input_video[-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif input_video[-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])

        masked_pred_list.append(masked_frame)
        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1
    out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    mask_archive = _archive_prediction_masks(io_args, video_name)

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], str(mask_archive)


def img_seq_type_input_tracking(SegTracker, io_args, video_name, imgs_path, fps, frame_num=0):

    pred_list = []
    masked_pred_list = []

    if frame_num > 0:
        output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
        output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])
        for i in range(0, frame_num):
            pred_list.append(np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
            masked_pred_list.append(cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        run_root = io_args['tracking_result_dir']
        _clear_output_directory(io_args['output_mask_dir'], run_root)
        _clear_output_directory(io_args['output_masked_frame_dir'], run_root)

    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])


    i_frame_num = frame_num

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        for img_path in imgs_path:
            if i_frame_num > 0:
                i_frame_num = i_frame_num - 1
                continue

            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            
            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask,seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, f'{frame_name}_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame,update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()
            
            save_prediction(pred_mask, output_mask_dir, f'{frame_name}.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx+frame_num, SegTracker.get_obj_num()),end='\r')
            frame_idx += 1
        print('\nfinished')
    
    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    height, width = pred_list[0].shape
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    i_frame_num =frame_num 

    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    for img_path in imgs_path:
        # if i_frame_num > 0:
        #     i_frame_num = i_frame_num - 1
        #     continue
        frame_name = os.path.basename(img_path).split('.')[0]
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png", masked_frame[:, :, ::-1])

        masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_name),end='\r')
        frame_idx += 1
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    mask_archive = _archive_prediction_masks(io_args, video_name)

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()


    return io_args['output_video'], str(mask_archive)
