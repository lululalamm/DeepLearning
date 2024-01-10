import numpy as np
from .json_utils import * 

"""
def _ConvertBoxToCOCOFormat(box):
  #Args:
  #  box: a [ymin, xmin, ymax, xmax] numpy array
  #Returns:
  #  a list of floats representing [xmin, ymin, width, height]
  
  return [float(box[1]), float(box[0]), float(box[3] - box[1]),
          float(box[2] - box[0])]
"""


def ExportSingleImageDetectionBoxesToCoco(image_id,
                                          category_id_set,
                                          detection_boxes,
                                          detection_scores,
                                          detection_classes,
                                          detection_keypoints=None,
                                          detection_keypoint_visibilities=None):
  """Export detections of a single image to COCO format.
  This function converts detections represented as numpy arrays to dictionaries
  that can be ingested by the COCO evaluation API. Note that the image_ids
  provided here must match the ones given to the
  ExporSingleImageDetectionBoxesToCoco. We assume that boxes, and classes are in
  correspondence - that is: boxes[i, :], and classes[i]
  are associated with the same groundtruth annotation.
  Args:
    image_id: unique image identifier either of type integer or string.
    category_id_set: A set of valid class ids. Detections with classes not in
      category_id_set are dropped.
    detection_boxes: float numpy array of shape [num_detections, 4] containing
      detection boxes.
    detection_scores: float numpy array of shape [num_detections] containing
      scored for the detection boxes.
    detection_classes: integer numpy array of shape [num_detections] containing
      the classes for detection boxes.
    detection_keypoints: optional float numpy array of keypoints
      with shape [num_detections, num_keypoints, 2].
    detection_keypoint_visibilities: optional integer numpy array of keypoint
      visibilities with shape [num_detections, num_keypoints]. Integer is
      treated as an enum with 0=not labels, 1=labeled but not visible and
      2=labeled and visible.
  Returns:
    a list of detection annotations for a single image in the COCO format.
  Raises:
    ValueError: if (1) detection_boxes, detection_scores and detection_classes
      do not have the right lengths or (2) if each of the elements inside these
      lists do not have the correct shapes or (3) if image_ids are not integers.
  """

  if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
    raise ValueError('All entries in detection_classes and detection_scores'
                     'expected to be of rank 1.')
  if len(detection_boxes.shape) != 2:
    raise ValueError('All entries in detection_boxes expected to be of '
                     'rank 2.')
  if detection_boxes.shape[1] != 4:
    raise ValueError('All entries in detection_boxes should have '
                     'shape[1] == 4.')
  num_boxes = detection_classes.shape[0]
  if not num_boxes == detection_boxes.shape[0] == detection_scores.shape[0]:
    raise ValueError('Corresponding entries in detection_classes, '
                     'detection_scores and detection_boxes should have '
                     'compatible shapes (i.e., agree on the 0th dimension). '
                     'Classes shape: %d. Boxes shape: %d. '
                     'Scores shape: %d' % (
                         detection_classes.shape[0], detection_boxes.shape[0],
                         detection_scores.shape[0]
                     ))
  detections_list = []
  for i in range(num_boxes):
    if detection_classes[i] in category_id_set:
      export_dict = {
          'image_id':
              image_id,
          'category_id':
              int(detection_classes[i]),
          'bbox':
              list(detection_boxes[i, :]),
          'score':
              float(detection_scores[i]),
      }
      if detection_keypoints is not None:
        keypoints = detection_keypoints[i]
        num_keypoints = keypoints.shape[0]
        if detection_keypoint_visibilities is None:
          detection_keypoint_visibilities = np.full((num_boxes, num_keypoints),
                                                    2)
        visibilities = np.reshape(detection_keypoint_visibilities[i], [-1])
        coco_keypoints = []
        for keypoint, visibility in zip(keypoints, visibilities):
          # Convert from [y, x] to [x, y] as mandated by COCO.
          coco_keypoints.append(float(keypoint[1]))
          coco_keypoints.append(float(keypoint[0]))
          coco_keypoints.append(int(visibility))
        export_dict['keypoints'] = coco_keypoints
        export_dict['num_keypoints'] = num_keypoints
      detections_list.append(export_dict)

  return detections_list

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def ExportDetectionsToCOCO(image_ids,
                           detection_boxes,
                           detection_scores,
                           detection_classes,
                           categories,
                           output_path=None):
  """Export detection annotations in numpy arrays to COCO API.
  This function converts a set of predicted detections represented
  as numpy arrays to dictionaries that can be ingested by the COCO API.
  Inputs to this function are lists, consisting of boxes, scores and
  classes, respectively, corresponding to each image for which detections
  have been produced.  Note that the image_ids provided here must
  match the ones given to the ExportGroundtruthToCOCO function in order
  for evaluation to work properly.
  We assume that for each image, boxes, scores and classes are in
  correspondence --- that is: detection_boxes[i, :], detection_scores[i] and
  detection_classes[i] are associated with the same detection.
  Args:
    image_ids: a list of unique image identifier either of type integer or
      string.
    detection_boxes: list of numpy arrays with shape [num_detection_boxes, 4]
    detection_scores: list of numpy arrays (float) with shape
      [num_detection_boxes]. Note that num_detection_boxes can be different
      for each entry in the list.
    detection_classes: list of numpy arrays (int) with shape
      [num_detection_boxes]. Note that num_detection_boxes can be different
      for each entry in the list.
    categories: a list of dictionaries representing all possible categories.
      Each dict in this list must have an integer 'id' key uniquely identifying
      this category.
    output_path: (optional) path for exporting result to JSON
  Returns:
    list of dictionaries that can be read by COCO API, where each entry
    corresponds to a single detection and has keys from:
    ['image_id', 'category_id', 'bbox', 'score'].
  Raises:
    ValueError: if (1) detection_boxes and detection_classes do not have the
      right lengths or (2) if each of the elements inside these lists do not
      have the correct shapes or (3) if image_ids are not integers.
  """
  category_id_set = set([cat['id'] for cat in categories])
  detections_export_list = []
  if not (len(image_ids) == len(detection_boxes) == len(detection_scores) ==
          len(detection_classes)):
    raise ValueError('Input lists must have the same length')
  for image_id, boxes, scores, classes in zip(image_ids, detection_boxes,
                                              detection_scores,
                                              detection_classes):
    if len(boxes.shape) != 2:
        print("no boxes")
        continue
    detections_export_list.extend(ExportSingleImageDetectionBoxesToCoco(
        image_id,
        category_id_set,
        boxes,
        scores,
        classes))
  if output_path:  
    with open(output_path, 'w') as fid:
      Dump(detections_export_list, fid, float_digits=4, indent=2, cls=NpEncoder)

  return detections_export_list