import argparse
import os
import time
from pathlib import Path

import cv2
import requests
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from numpy import random
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, increment_path,
                           non_max_suppression_lmks, scale_coords,
                           scale_coords_lmks, set_logging, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel, select_device, time_synchronized

API = os.getenv("API_LINK")
PORT = os.getenv("API_PORT")


def sendImage(frame):
    response = 'UNK'
    try:
        imencoded = cv2.imencode(".jpg", frame)[1]
        file = {'img': ("image.jpg", imencoded.tobytes(), "image/jpeg")}
        response = requests.post(
            'http://' + API + ':' + PORT+'/predict', files=file)
        response = response.content
    except Exception as e:
        pass
    return response


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
        not opt.no_trace,
    )
    save_img = not opt.nosave and not source.endswith(
        ".txt")  # save inference images
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(
                device).type_as(next(model.parameters()))
        )  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression_lmks(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms
        )
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(
                ), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                det[:, [6, 7, 9, 10, 12, 13, 15, 16, 18, 19]] = scale_coords_lmks(
                    img.shape[2:], det[:, [6, 7, 9, 10,
                                           12, 13, 15, 16, 18, 19]], im0.shape
                ).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for det_per_box in reversed(det):
                    xyxy, conf, cls, lmks, lmks_mask = (
                        det_per_box[0:4],
                        det_per_box[4],
                        det_per_box[5],
                        det_per_box[[6, 7, 9, 10, 12, 13, 15, 16, 18, 19]],
                        det_per_box[[8, 11, 14, 17, 20]],
                    )

                    # print(xyxy)

                    frame = im0[int(xyxy[1]) - 10:int(xyxy[3] + 10),
                                int(xyxy[0]) - 10:int(xyxy[2]) + 10]
                    identity = sendImage(frame)

                    if save_img or view_img:  # Add bbox to image
                        plot_one_box(
                            xyxy,
                            im0,
                            label=str(identity),
                            color=colors[int(cls)],
                            line_thickness=1,
                            lmks=lmks,
                            lmks_mask=(lmks_mask > 0.5).float(),
                            lmks_normalized=False,
                            radius=4,
                        )

            # Print time (inference + NMS)
            print(
                f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            )

            # Stream results
            if view_img:
                cv2.imwrite(str(p)+".jpg", im0)
                im0 = cv2.resize(im0, (int(640*1.5), int(480*1.5)))
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov7-tiny76.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="0", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument("--img-size", type=int, default=640,
                        help="inference size (pixels)")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument("--iou-thres", type=float,
                        default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--device", default="0",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true",
                        help="display results")
    parser.add_argument("--save-txt", action="store_true",
                        help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument("--nosave", action="store_true",
                        help="do not save images/videos")
    parser.add_argument(
        "--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3"
    )
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--update", action="store_true",
                        help="update all models")
    parser.add_argument("--project", default="runs/detect",
                        help="save results to project/name")
    parser.add_argument("--name", default="exp",
                        help="save results to project/name")
    parser.add_argument(
        "--exist-ok", action="store_true", help="existing project/name ok, do not increment"
    )
    parser.add_argument("--no-trace", action="store_true",
                        help="don`t trace model")
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7-tiny76.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
