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
    0: "person",
    1: "forklift",
    2: "hardhat",
    3: "safety_vest",
    4: "cone",

