# Test fixtures

`detection_events.TEST_FIXTURE.jsonl` is **hand-written test data** in Sentinel's internal detection-event format (`schema/detection_event.schema.json`).

- It is **not** output from PipeGen, TensorRT, DeepStream or any real detector.
- The boxes, labels and confidences are invented, so unit tests can exercise parsing, validation and frame association.
- It must never be presented as evidence of a detector or PipeGen run.
