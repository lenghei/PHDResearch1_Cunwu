"""Profile missing FM_D inference settings on all 548 Normal validation images."""
import argparse
from copy import deepcopy
import csv
import gc
from pathlib import Path
import os
from statistics import mean, median
import sys
from time import perf_counter

from validate_checkpoints import (CACHE, CACHE_SHA, MAIN, HERE, PROTOCOL, VIS_NAMES, catalog,
                                  digest, healthy_state, image_from_cache, input_set, load,
                                  output_folder, percentile95, require, source_files, write_json)
from model import declared_profile_configuration

BASE = "yolov8s_cft640"
YOLO26 = "ck01_yolo26s_ft1280"
SAHI = MAIN.parent / "datasets/cache/uav03_sahi_01136"
BUDGET = 1200
CASES = {
    "v8_full640": (BASE, 640, "FullFrame", "NMS", "None", False),
    "v8_full1536": (BASE, 1536, "FullFrame", "NMS", "None", False),
    "v8_sahi640": (BASE, 640, "SAHI_Slicing", "NMS", "None", False),
    "v8_tta640": (BASE, 640, "FullFrame", "NMS", "None", True),
    "v8_clahe640": (BASE, 640, "FullFrame", "NMS", "CLAHE_Pre", False),
    "v8_gamma640": (BASE, 640, "FullFrame", "NMS", "Gamma_Pre", False),
    "v8_denoise640": (BASE, 640, "FullFrame", "NMS", "Denoise_Pre", False),
    "v8_deblur640": (BASE, 640, "FullFrame", "NMS", "Deblur_Pre", False),
    "v26_dense640": (YOLO26, 640, "FullFrame", "NMS", "None", False),
    "v26_e2e640": (YOLO26, 640, "FullFrame", "EndToEnd_NMSFree", "None", False),
}
V11 = "ck06_yolo11s_ft1280"
V8N = "yolov8n_ft640"
V26N = "ck04_yolo26n_native10_full20"
ADDITIONAL_CASES = {
    "v11_full640": (V11, 640, "FullFrame", "NMS", "None", False),
    "v11_fp16_640": (V11, 640, "FullFrame", "NMS", "None", False),
    "v11_full1536": (V11, 1536, "FullFrame", "NMS", "None", False),
    "v11_sahi640": (V11, 640, "SAHI_Slicing", "NMS", "None", False),
    "v11_tta640": (V11, 640, "FullFrame", "NMS", "None", True),
    "v11_clahe640": (V11, 640, "FullFrame", "NMS", "CLAHE_Pre", False),
    "v8n_full640": (V8N, 640, "FullFrame", "NMS", "None", False),
    "v8n_denoise640": (V8N, 640, "FullFrame", "NMS", "Denoise_Pre", False),
    "v8n_deblur640": (V8N, 640, "FullFrame", "NMS", "Deblur_Pre", False),
    "v26n_dense640": (V26N, 640, "FullFrame", "NMS", "None", False),
    "v26n_e2e640": (V26N, 640, "FullFrame", "EndToEnd_NMSFree", "None", False),
}
ALL_CASES = {**CASES, **ADDITIONAL_CASES}
GROUPS = {
    "yolo11s": tuple(name for name in ADDITIONAL_CASES if name.startswith("v11_")),
    "yolov8n": tuple(name for name in ADDITIONAL_CASES if name.startswith("v8n_")),
    "yolo26n-nms": ("v26n_dense640",),
    "yolo26n-e2e": ("v26n_e2e640",),
}
PRECISION = {"v11_fp16_640": "FP16"}
MODEL_FEATURES = {
    BASE: ("YOLOv8", "Small", "Train640"),
    YOLO26: ("YOLO26", "Small", "Train1280"),
    V11: ("YOLO11", "Small", "Train1280"),
    V8N: ("YOLOv8", "Nano", "Train640"),
    V26N: ("YOLO26", "Nano", "Train640"),
}
SETTINGS = {"CLAHE_Pre": "LAB L; clip 2, tiles 8x8", "Gamma_Pre": "gamma 0.8, direct exponent",
            "Denoise_Pre": "Gaussian 3x3, sigma 0.8", "Deblur_Pre": "unsharp proxy; 5x5 sigma 1.2, amount 0.6",
            "SAHI_Slicing": "640px tiles; 20% overlap; clamped to image; no full-image pass; class-aware global NMS .7",
            "TTA": "native Ultralytics DetectionModel augmented forward"}


def case_spec(name):
    checkpoint, size, strategy, post, pre, tta = ALL_CASES[name]
    arch, scale, resolution = MODEL_FEATURES[checkpoint]
    return declared_profile_configuration(arch, scale, "VisDrone_FT", resolution, size,
                                          PRECISION.get(name, "FP32"),
                                          strategy, post, tta=tta, preprocessing=pre)


def transform(name):
    import cv2
    import numpy as np

    if name == "CLAHE_Pre":
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        def apply(pixels):
            lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
            lab[...,0] = clahe.apply(lab[...,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return apply
    if name == "Gamma_Pre":
        lut = np.rint((np.arange(256)/255)**.8*255).astype("uint8")
        return lambda img: cv2.LUT(img,lut)
    if name == "Denoise_Pre":
        return lambda img: cv2.GaussianBlur(img,(3,3),.8)
    if name == "Deblur_Pre":
        return lambda img: cv2.addWeighted(img,1.6,cv2.GaussianBlur(img,(5,5),1.2),-.6,0)
    return lambda img: img


def read_models():
    return {item["checkpoint"]: item for item in catalog() if item["checkpoint"] in MODEL_FEATURES}


def profile_case(out, cache, images, name, entry, modules):
    torch, YOLO, basic, traced, get_slices, batched_nms = modules
    checkpoint, size, strategy, post, pre, tta = ALL_CASES[name]
    precision = PRECISION.get(name, "FP32")
    half = precision == "FP16"
    e2e = post == "EndToEnd_NMSFree"
    tiled = strategy == "SAHI_Slicing"
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    require(before == 0, "Live tensors remained from the preceding case.")
    model = YOLO(source_files(entry))
    require(list(model.names.values()) == VIS_NAMES, "Checkpoint is not native ten.")
    healthy_state(model.model, torch)
    network_shapes = []
    handle = model.model.model[0].register_forward_pre_hook(
        lambda module, args: network_shapes.append(list(args[0].shape)))
    preprocess = transform(pre)

    def detect(pixels):
        with torch.inference_mode():
            result, = model.predict(pixels, predictor=basic if e2e else traced, imgsz=size, batch=1,
                device="0", half=half, conf=PROTOCOL["conf"], iou=PROTOCOL["iou"],
                max_det=PROTOCOL["max_det"], end2end=e2e, rect=False, agnostic_nms=False,
                augment=tta, save=False, verbose=False)
        require(result.orig_shape == pixels.shape[:2], "Output coordinate frame differs.")
        return result.boxes.data

    def infer(pixels):
        network_shapes.clear()
        frame = preprocess(pixels)
        require(frame.shape == pixels.shape and frame.dtype == pixels.dtype, "Preprocessing changed geometry.")
        if not tiled:
            boxes = detect(frame)
        else:
            parts = []
            height, width = frame.shape[:2]
            for x0,y0,x1,y1 in get_slices(height, width, slice_height=640, slice_width=640,
                    overlap_height_ratio=.2, overlap_width_ratio=.2, auto_slice_resolution=False):
                selected = detect(frame[y0:y1, x0:x1]).clone()
                selected[:, :4] += selected.new_tensor([x0,y0,x0,y0])
                parts.append(selected)
            candidates = torch.cat(parts)
            if len(candidates)>30000:
                candidates = candidates[candidates[:,4].argsort(descending=True)[:30000]]
            kept = batched_nms(candidates[:,:4], candidates[:,4], candidates[:,5], .7)[:3000]
            boxes = candidates[kept]
        torch.cuda.synchronize()
        return boxes, network_shapes[:]

    torch.cuda.empty_cache()
    started = perf_counter()
    samples = []
    try:
        warm = image_from_cache(cache, images[0])
        for _ in range(3):
            infer(warm)
        require(bool(model.predictor.model.end2end) == e2e and bool(model.predictor.args.augment) == tta,
                "Backend did not enable the requested inference mode.")
        require(bool(model.predictor.model.fp16) == half, "Backend precision differs.")
        if e2e:
            head = model.model.model[-1]
            require(head.max_det == 3000, "Unexpected internal end-to-end detection cap.")
        torch.cuda.reset_peak_memory_stats()
        pass_started = perf_counter()
        for row in images:
            clock = perf_counter()
            pixels = image_from_cache(cache, row)
            decoded = perf_counter()
            boxes, shapes = infer(pixels)
            predicted = perf_counter()
            values = boxes.detach().cpu().numpy()
            from numpy import isfinite
            require(isfinite(values).all() and (values[:,5]>=0).all() and (values[:,5]<10).all(),
                    "Invalid native-ten predictions.")
            finished = perf_counter()
            require(all(s[:2]==[1,3] and s[2]>0 and s[3]>0 for s in shapes)
                    and (len(shapes)>1 if tta else bool(shapes))
                    and (tta or all(s == [1,3,size,size] for s in shapes))
                    and (tta or tiled or len(shapes)==1), "Network branch trace differs.")
            samples.append({"image_id": row["image_id"], "network_shapes": shapes, "detections":len(values),
                            "predict_ms":1000*(predicted-decoded), "pipeline_ms":1000*(finished-clock)})
        summary = {"case":name,"checkpoint":checkpoint,"checkpoint_sha256":entry["sha256"],"images":len(samples),
                   "resolution":size,"strategy":strategy,"postprocessing":post,"precision":precision,
                   "preprocessing":pre,"TTA":tta,"parameters":int(entry["parameters"]),
                   "FM_D_configuration_id":case_spec(name)["configuration_id"],
                   "network_forwards_mean":mean(len(s["network_shapes"]) for s in samples),
                   "warmup_seconds":pass_started-started, "pass_wall_seconds":perf_counter()-pass_started,
                   "allocated_before_case_bytes":before,
                   "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                   "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}
        for field in ("predict","pipeline"):
            values = [r[f"{field}_ms"] for r in samples]
            summary.update({f"{field}_mean_ms":mean(values),f"{field}_median_ms":median(values),
                            f"{field}_p95_ms":percentile95(values)})
        write_json(out/(name+".json"),{"summary":summary,"samples":samples})
        print(f"{name}: {len(samples)} images | pipeline {summary['pipeline_mean_ms']:.2f} ms/image", flush=True)
        return summary,samples
    finally:
        handle.remove()
        model.predictor = None
        model.model = None


def operations(summary, samples, entry, torch, YOLO, profiles):
    from thop import profile

    total = 0
    for sample in samples:
        for shape in sample["network_shapes"]:
            key = (summary["checkpoint"], summary["postprocessing"],*shape)
            if key not in profiles:
                net = deepcopy(YOLO(source_files(entry)).model).cpu().float().eval()
                net.end2end = summary["postprocessing"] == "EndToEnd_NMSFree"
                if net.end2end:
                    net.set_head_attr(max_det=3000, agnostic_nms=False)
                net.fuse(verbose=False)
                with torch.inference_mode():
                    macs,_ = profile(net,inputs=(torch.zeros(*shape),),verbose=False)
                require(macs>0,"Operation count was not measured.")
                profiles[key]=2*macs/1e9
            total += profiles[key]
    return total/len(samples)


def worker(out, cache, names):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from torchvision.ops import batched_nms
    from main.predictor import IEEEDetectionPredictor
    sys.path.insert(0,str(SAHI.resolve(strict=True)))
    from sahi.slicing import get_slice_bboxes
    from importlib.metadata import version
    require(ultralytics.__version__ == "8.4.61" and version("sahi") == "0.11.36"
            and torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] >= 3*2**30,
            "Frozen runtime or GPU unavailable.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, images, _, _ = input_set(cache,cache/"ground_truth.json")
    catalog_rows = read_models()
    write_json(out/"environment.json",{"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
               "cuda":torch.version.cuda,"ultralytics":ultralytics.__version__,"sahi":version("sahi")})
    all_rows, raw = [], []
    for name in names:
        entry = catalog_rows[ALL_CASES[name][0]]
        row,samples = profile_case(out,cache,images,name,entry,(torch,YOLO,DetectionPredictor,
                                   IEEEDetectionPredictor,get_slice_bboxes,batched_nms))
        all_rows.append(row)
        raw.append(samples)
    profiles = {}
    for row,samples in zip(all_rows,raw):
        row["mean_GFLOPs_2xMAC"] = operations(row,samples,catalog_rows[row["checkpoint"]],torch,YOLO,profiles)
    write_json(out/"operation_profiles.json", [{"binding":list(k),"GFLOPs_2xMAC":v} for k,v in profiles.items()])
    with (out/"inference_options.csv").open("x",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(out/"result.json",{"status":"COMPLETE","RunID":out.name,"cases":len(names),
               "images_per_case":548,"image_calls":548*len(names),"scope":"DEVELOPMENT_RESOURCE_ONLY"})


def main():
    from supervisor import guarded_run, sample

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=CACHE)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case",action="append",choices=ALL_CASES,
                           help="Defaults to the original nine cases. NMS-free cases run alone.")
    selection.add_argument("--group", choices=GROUPS, help="Run a matched extension group.")
    parser.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    requested = GROUPS[args.group] if args.group else args.case
    if not requested:
        requested = [name for name in CASES if name != "v26_e2e640"]
    names = [name for name in ALL_CASES if name in requested]
    if len(names) > 1 and any(ALL_CASES[name][3] == "EndToEnd_NMSFree" for name in names):
        parser.error("Run NMS-free cases in separate processes to isolate GPU memory.")
    cache=args.cache.resolve(strict=True)
    if args.worker:
        out=args.worker.resolve(strict=True)
        require(os.environ.get("UAVGAP_INFER_OPTIONS")==str(out)
                and load(out/"plan.json")["order"]==names,"Run must be supervised and match the plan.")
        worker(out,cache,names)
        return
    lock = MAIN.parent / "datasets/local_runs/uav_05_evaluation/.gpu.lock"
    if lock.exists():
        parser.exit(2, f"GPU reserved by another experiment: {lock.read_text(encoding='utf-8')}\n")
    input_set(cache,cache/"ground_truth.json")
    out=output_folder(HERE.name)
    print(out,flush=True)
    runtime=out/"runtime"
    (runtime/"Ultralytics").mkdir(parents=True)
    plan={"order":names,"cases":{name:case_spec(name) for name in names},"parameters":SETTINGS,
          "scope":"DEVELOPMENT_RESOURCE_ONLY","members_sha256":CACHE_SHA,"images":548,"warmups":3,
          "INT8":"NOT_ADMITTED_NO_VERIFIED_CALIBRATED_BACKEND"}
    write_json(out/"plan.json", plan)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(out),"--cache",str(cache)]
    for name in names:
        command.extend(("--case",name))
    env=dict(os.environ, UAVGAP_INFER_OPTIONS=str(out), YOLO_AUTOINSTALL="false",
             YOLO_OFFLINE="true", YOLO_CONFIG_DIR=str(runtime), PYTHONDONTWRITEBYTECODE="1")
    write_json(out/"execution.json",{"command":command,"budget_seconds":BUDGET,
               "code_sha256":digest(Path(__file__)),"sahi_slicing_sha256":digest(SAHI/"sahi/slicing.py")})
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(out))
    try:
        guarded_run(command,out,BUDGET,env,sample()["uuid"])
        require(load(out/"result.json")["status"]=="COMPLETE","Incomplete resource measurement.")
    finally:
        lock.unlink()


if __name__=="__main__":
    main()
