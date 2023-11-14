import torch
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression,Profile,scale_boxes,xyxy2xywh
from pathlib import Path
from utils.plots import Annotator
import cv2
from PIL import Image
from torchvision.transforms import ToTensor,Resize
def get_yolo_results(image_path,model):
    source = image_path
    imgsz=(640, 640)
    stride = model.stride
    pt = model.pt
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    conf_thres = 0.25
    iou_thres=0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    webcam = False
    save_dir = Path('D:/Upwork/KS/MOHIT/')
    save_crop = False
    line_thickness = 3
    names = model.names

    total_detections = []
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = False
                augment = False
                pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                txt_path = str(save_dir / p.stem)
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    save_conf = True
                    for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
    #                         with open(f'{txt_path}.txt', 'a') as f:
    #                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            total_detections.append(line)
        
    converted_detections = [tuple(value.item() if isinstance(value, torch.Tensor) else value for value in item) for item in total_detections]
        
    return converted_detections

def get_bboxes(image_path, detections):
    image = cv2.imread(image_path)
    
    bounding_boxes = []
    
    for cls, x, y, w, h, confidence in detections:
        cls = int(cls)
        left = int((x - w/2) * image.shape[1]) - 10
        top = int((y - h/2) * image.shape[0]) - 10
        width = int(w * image.shape[1]) + 20
        height = int(h * image.shape[0]) + 20
        
        bounding_boxes.append({
            'class': cls,
            'left': left,
            'top': top,
            'width': width,
            'height': height
        })
    
    return bounding_boxes

def calculate_bbox_area(bbox):
    width = bbox['width']
    height = bbox['height']
    return width * height

def remove_lowest_area_bbox(detections):
    if len(detections) <= 2:
        return detections

    areas = [calculate_bbox_area(bbox) for bbox in detections]
    min_area_index = areas.index(min(areas))
    
    # Return detections without the detection with the lowest area
    return detections[:min_area_index] + detections[min_area_index + 1:]
def perform_inference(model, image, model_args):
    transform = ToTensor()
    resize = resize = Resize((32, 128))   # Resize the image to match the model's input size
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = resize(image)  # Resize the image
    image_tensor = transform(image).unsqueeze(0).to(model_args['device'])

    with torch.no_grad():
        output = model(image_tensor)
        # Process the output as needed

    return output

def ocr_text(model,image, model_args,loaded_tokenizer):
    inference_result = perform_inference(model, image, model_args)
    # Greedy decoding
    pred = inference_result.softmax(-1)
    #label, confidence = model.tokenizer.decode(pred)
    label, confidence = loaded_tokenizer.decode(pred)
    return (label[0],["{:.2%}".format(value) for value in confidence[0].tolist()[:-1]])

def draw_detections_with_text(image, detections):
    font_scale = 3 * (image.shape[1]//1000)
    thickness_text = 10 * (image.shape[1]//1000)
    print(thickness_text)
    dot_peen_detection = detections['DotPeenText']
    left = dot_peen_detection[1]['left']
    top = dot_peen_detection[1]['top']
    width = dot_peen_detection[1]['width']
    height = dot_peen_detection[1]['height']
    text = dot_peen_detection[0][0]
    # Draw the bbox on the image
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2  # Thickness of the bbox lines
    cv2.rectangle(image, (left, top), (left + width, top + height), color, thickness)
    cv2.putText(image, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness_text)
    
    machine_peen_detection = detections['MachinePeenText']
    left = machine_peen_detection[1]['left']
    top = machine_peen_detection[1]['top']
    width = machine_peen_detection[1]['width']
    height = machine_peen_detection[1]['height']
    text = machine_peen_detection[0][0]
    # Draw the bbox on the image
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2  # Thickness of the bbox lines
    cv2.rectangle(image, (left, top), (left + width, top + height), color, thickness)
    cv2.putText(image, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness_text) 
    return image

def process_detections(detections,image_path,model_ocr, model_args,loaded_tokenizer):
    results ={}
    results['DotPeenText'] = []
    results['MachinePeenText'] = []
    img = cv2.imread(image_path)
    for detection in detections:
        if detection['class'] == 0:
            cropped_img = img[detection['top']:detection['top']+detection['height'],detection['left']:detection['left']+detection['width']]
            results['DotPeenText'].append(ocr_text(model_ocr,cropped_img, model_args,loaded_tokenizer))
            dot_peen_detection = [d for d in detections if d['class'] == 0][0]
            results['DotPeenText'].append(dot_peen_detection)
        else:
            cropped_img = img[detection['top']:detection['top']+detection['height'],detection['left']:detection['left']+detection['width']]
            results['MachinePeenText'].append(ocr_text(model_ocr,cropped_img, model_args,loaded_tokenizer))
            machine_peen_detection = [d for d in detections if d['class'] == 1][0]
            results['MachinePeenText'].append(machine_peen_detection)
    return results