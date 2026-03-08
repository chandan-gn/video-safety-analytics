# AI Engineer — Video Analytics Take-Home Assignment
## Building a Safety Monitoring Pipeline

The platform should ingest video stream, runs file-tuned YOLO-based object detection, tracks objects across frames.
Evaluates configured safety rules and publishes if any violations are found. 


The final pipeline should have

Video Ingestion
Object Detection
Tracking objects
Violation detection
Publishing Events
Observing pipeline (prometheus)


Classes mapped
{
    0: "person",
    1: "hard-hat",
    2: "gloves",
    3: "mask",
    4: "glasses",
    5: "boots",
    6: "vest",
    7: "ppe-suit",
    8: "ear-protector",
    9: "safety-harness",
    10: "no-hard-hat",
    11: "no-gloves",
    12: "no-mask",
    13: "no-glasses",
    14: "no-boots",
    15: "no-vest",
    16: "no-ppe-suit",
}

The YOLO base model used is yolo8m.pt is fine-tuned on SH17 dataset. The fine-tuned model is imported in ONNX format(best.onnx).

To run the pipeline, start server
uvicorn api:app --reload

1.) In CLI,
python main.py --source VIDEO.mp4

or

2.) Using curl,
curl -X POST http://127.0.0.1:8000/run -F "file=@video.mp4"

Note that the filename should be given after @

