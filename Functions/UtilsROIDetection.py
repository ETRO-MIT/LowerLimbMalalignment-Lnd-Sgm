# Python packages
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Custom functions
from Functions.Utils import transform_coordinates, flatten


# Function to get the Faster R-CNN model from the TorchVision repository
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Filter and get the first 8 bboxes from the neural network
def filter_estimations(set_of_estimations):
    # Get first two femur head bboxes
    index_femur = (set_of_estimations[0]['labels'] == 1).nonzero(as_tuple=True)
    index_femur = index_femur[0][:2]
    femur_bboxes = set_of_estimations[0]['boxes'][index_femur]

    # Get the first two diaphysis bboxes
    index_diaphysis = (set_of_estimations[0]['labels'] == 2).nonzero(as_tuple=True)
    index_diaphysis = index_diaphysis[0][:2]
    diaphysis_bboxes = set_of_estimations[0]['boxes'][index_diaphysis]

    # Get first two knee bboxes
    index_knee = (set_of_estimations[0]['labels'] == 3).nonzero(as_tuple=True)
    index_knee = index_knee[0][:2]
    knee_bboxes = set_of_estimations[0]['boxes'][index_knee]

    # Get first two ankle bboxes
    index_ankle = (set_of_estimations[0]['labels'] == 4).nonzero(as_tuple=True)
    index_ankle = index_ankle[0][:2]
    ankle_bboxes = set_of_estimations[0]['boxes'][index_ankle]

    bboxes = [femur_bboxes[0], femur_bboxes[1], diaphysis_bboxes[0], diaphysis_bboxes[1],
              knee_bboxes[0], knee_bboxes[1], ankle_bboxes[0], ankle_bboxes[1]]

    filtered_bboxes = []
    for box in bboxes:
        box_numpy = box.cpu().numpy()
        filtered_bboxes.append([int(box_numpy[0]), int(box_numpy[1]), int(box_numpy[2]), int(box_numpy[3])])

    return filtered_bboxes


# Transform bounding-box coordinates to their original dimensions
def transform_bboxes(original_image, target_image, set_of_bboxes):
    new_set_bboxes = []
    for bbox in set_of_bboxes:
        coordinates = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
        new_coordinates = flatten(transform_coordinates(original_image, target_image, coordinates))
        new_set_bboxes.append(new_coordinates)

    return new_set_bboxes


# Arrange the estimated boxes so that they follow the right/left leg format  estimations
def get_right_left_bboxes(bbox1, bbox2):
    x1 = bbox1[0]
    x2 = bbox2[0]
    half_value = (x1 + x2) / 2
    if x1 < half_value:
        right_bbox = bbox1
        left_bbox = bbox2
        return right_bbox, left_bbox
    else:
        right_bbox = bbox2
        left_bbox = bbox1
        return right_bbox, left_bbox


# Arrange each of the components of the estimations
def arrange_bboxes(set_of_boxes):
    right_femur_box, left_femur_box = get_right_left_bboxes(set_of_boxes[0], set_of_boxes[1])
    right_diaphysis_box, left_diaphysis_box = get_right_left_bboxes(set_of_boxes[2], set_of_boxes[3])
    right_knee_box, left_knee_box = get_right_left_bboxes(set_of_boxes[4], set_of_boxes[5])
    right_ankle_box, left_ankle_box = get_right_left_bboxes(set_of_boxes[6], set_of_boxes[7])

    arranged_bboxes = [right_femur_box, left_femur_box, right_diaphysis_box, left_diaphysis_box,
                       right_knee_box, left_knee_box, right_ankle_box, left_ankle_box]

    return arranged_bboxes
